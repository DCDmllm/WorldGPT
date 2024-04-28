import logging
import os.path
import random
from collections import OrderedDict
from typing import List

import torch
from torch.nn.utils import rnn
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList, LlamaTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from .modeling_llama import LlamaForCausalLM
from .reflector import Reflector
from .layers import *


def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Args:
        u: (N, T_I_V_A.txt, D) tensor.
        v: (N, T_I_V_A.txt, D) tensor.
    Returns:
        l1_loss: (N,) tensor of summed L1 loss.
    """
    assert u.shape == v.shape, (u.shape, v.shape)
    return ((u - v) ** 2).sum(dim=-1) ** 0.5


def load_worldgpt_model(**args):
    model = WorldGPTModel(**args)
    if 'load_path' in args:
        delta_ckpt = torch.load(os.path.join(args['load_path'], 'pytorch_model.pt'), map_location=torch.device('cuda'))
        checkpoint = OrderedDict()
        for k, v in delta_ckpt.items():
            if 'llama_model.model.embed_tokens.weight' in k:
                checkpoint['llama_model.base_model.model.model.embed_tokens.weight'] = v
            elif 'llama_model.lm_head.weight' in k:
                checkpoint['llama_model.base_model.model.lm_head.weight'] = v
            else:
                checkpoint[k] = v
        model.load_state_dict(checkpoint, strict=False)
    return model


class WorldGPTModel(nn.Module):
    """LoRA for LLaMa model"""

    def __init__(self, **args):
        super(WorldGPTModel, self).__init__()
        self.args = args

        self.max_length = args['max_length']
        self.device = torch.cuda.current_device()
        print('args max_length', args['max_length'])

        self.vicuna_ckpt_path = self.args['vicuna_path']
        print(f'Initializing language decoder from {self.vicuna_ckpt_path} ...')

        self.llama_model = LlamaForCausalLM.from_pretrained(self.vicuna_ckpt_path, low_cpu_mem_usage=True, torch_dtype=args['dtype'])
        if self.args.get('enable_lora'):
            print("Adding LoRa module ...")
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.args['lora_r'],
                lora_alpha=self.args['lora_alpha'],
                lora_dropout=self.args['lora_dropout'],
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
            )

            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        else:
            print("Disabling LoRa ...")
            for param in self.llama_model.parameters():
                param.requires_grad = False
            self.llama_model.eval()
        print('Language decoder initialized.')

        # use the new trained tokenizer
        tokenizer_path = self.vicuna_ckpt_path
        print(f'Initializing tokenizer from {tokenizer_path} ...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        # self.llama_tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self.args['gen_token_idx'] = {}
        self._add_image_token()
        self._add_video_token()
        self._add_audio_token()
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        print('Tokenizer initialized.')
        
        self.llama_proj = nn.Linear(768, self.llama_model.config.hidden_size)
        if self.args.get('freeze_input_proj'):
            for param in self.llama_proj.parameters():
                param.requires_grad = False

        self.input_embeddings = self.llama_model.get_input_embeddings()

        if self.args['enable_decode']:
            self.gen_image_hidden_fc = TextFcLayer(self.llama_model.config.hidden_size, self.args['image_output_dim'],
                                                  num_input_tokens=self.args['num_gen_img_tokens'],
                                                  num_output_tokens=self.args['image_num_output_tokens'])
            self.gen_video_hidden_fc = TextFcLayer(self.llama_model.config.hidden_size, self.args['video_output_dim'],
                                                  num_input_tokens=self.args['num_gen_img_tokens'],
                                                  num_output_tokens=self.args['video_num_output_tokens'])
            self.gen_audio_hidden_fc = TextFcLayer(self.llama_model.config.hidden_size, self.args['audio_output_dim'],
                                                  num_input_tokens=self.args['num_gen_img_tokens'],
                                                  num_output_tokens=self.args['audio_num_output_tokens'])
            if self.args.get('freeze_output_proj'):
                self.gen_image_hidden_fc.requires_grad_(False)
                self.gen_video_hidden_fc.requires_grad_(False)
                self.gen_audio_hidden_fc.requires_grad_(False)

        self.gen_text_hidden_fc = TextFcLayer(self.llama_model.config.hidden_size, 768,
                                              num_input_tokens=self.args['num_gen_img_tokens'],
                                              num_output_tokens=1)
        if self.args.get('freeze_output_proj'):
            self.gen_text_hidden_fc.requires_grad_(False)
        
        if self.args.get('enable_reflector'):
            self.reflector = Reflector(768, self.llama_model.config.hidden_size,
                                       num_output_tokens=self.args['num_reflector_tokens'])
            # self.llama_tokenizer.add_tokens(["<Ref>"])  # add special reflector token to tokenizer
            # self.llama_tokenizer.add_tokens(["</Ref>"])  # add special reflector token to tokenizer
            if self.args.get('freeze_reflector'):
                self.reflector.requires_grad_(False)
        
        self.lora_params = []
        if self.args.get('enable_lora'):
            for k, v in self.llama_model.named_parameters():
                if v.requires_grad:
                    self.lora_params.append(f'llama_model.{k}')
                    if self.args.get('freeze_llm'):
                        v.requires_grad = False
        

    def _add_image_token(self):
        # Add an image token for loss masking (and visualization) purposes.
        self.llama_tokenizer.add_tokens(["<Img>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["</Img>"])  # add special image token to tokenizer

        # Add [IMG] tokens to the vocabulary.
        self.args['gen_token_idx']['image'] = []
        for i in range(self.args['num_gen_img_tokens']):
            print(f'Adding [IMG{i}] token to vocabulary.')
            print(f'Before adding new token, tokenizer("[IMG{i}]") =',
                  self.llama_tokenizer(f'[IMG{i}]', add_special_tokens=False))
            num_added_tokens = self.llama_tokenizer.add_tokens(f'[IMG{i}]')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("[IMG{i}]") =',
                  self.llama_tokenizer(f'[IMG{i}]', add_special_tokens=False))
            gen_token_idx = self.llama_tokenizer(f'[IMG{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_token_idx']['image'].append(gen_token_idx[0])

    def _add_video_token(self):
        self.llama_tokenizer.add_tokens({"<Vid>"})  # add special video token to tokenizer
        self.llama_tokenizer.add_tokens({"</Vid>"})  # add special video token to tokenizer

        # Add [VID] tokens to the vocabulary.
        self.args['gen_token_idx']['video'] = []
        for i in range(self.args['num_gen_img_tokens']):
            print(f'Adding [VID{i}] token to vocabulary.')
            print(f'Before adding new token, tokenizer("[VID{i}]") =',
                  self.llama_tokenizer(f'[VID{i}]', add_special_tokens=False))
            num_added_tokens = self.llama_tokenizer.add_tokens(f'[VID{i}]')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("[VID{i}]") =',
                  self.llama_tokenizer(f'[VID{i}]', add_special_tokens=False))
            gen_token_idx = self.llama_tokenizer(f'[VID{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_token_idx']['video'].append(gen_token_idx[0])

    def _add_audio_token(self):
        self.llama_tokenizer.add_tokens({"<Aud>"})  # add special audio token to tokenizer
        self.llama_tokenizer.add_tokens({"</Aud>"})  # add special audio token to tokenizer

        # Add [AUD] tokens to the vocabulary.
        self.args['gen_token_idx']['audio'] = []
        for i in range(self.args['num_gen_img_tokens']):
            print(f'Adding [AUD{i}] token to vocabulary.')
            print(f'Before adding new token, tokenizer("[AUD{i}]") =',
                  self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False))
            num_added_tokens = self.llama_tokenizer.add_tokens(f'[AUD{i}]')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("[AUD{i}]") =',
                  self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False))
            gen_token_idx = self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_token_idx']['audio'].append(gen_token_idx[0])

    def _llama_embeds(self, input_ids):
        if self.args.get('enable_lora'):
            return self.llama_model.model.model.embed_tokens(input_ids)  # bsz x len x llama_size
        else:
            return self.llama_model.model.embed_tokens(input_ids)  # bsz x len x llama_size
    
    def _output_proj(self, enable_decode, modality, *args, **kwargs):
        if enable_decode:
            if modality == 'image':
                return self.gen_image_hidden_fc(*args, **kwargs)
            elif modality == 'video':
                return self.gen_video_hidden_fc(*args, **kwargs)
            elif modality == 'audio':
                return self.gen_audio_hidden_fc(*args, **kwargs)
        else:
            return self.gen_text_hidden_fc(*args, **kwargs)

    def _get_p_before(self, batch_size):
        """
        [BOS]### Human: 
        """
        bos = torch.ones([batch_size, 1], dtype=torch.long,
                         device=self.device) * self.llama_tokenizer.bos_token_id  # bsz x 1
        p_before = '### Human: '
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(
                self.device).input_ids
        bos_embeds = self._llama_embeds(bos)  # bsz x 1 x llama_size
        p_before_embeds = self._llama_embeds(p_before_tokens).expand(batch_size, -1, -1)  # bsz x s x llama_size
        bos_p_before_embeds = torch.cat([bos_embeds, p_before_embeds], dim=1)  # bsz x (1 (bos) + s) x llama_size

        targets_before = torch.ones([batch_size, bos_p_before_embeds.size()[1]],  # bsz x (1 (bos) + s)
                                     dtype=torch.long).to(self.device).fill_(-100)
        attention_mask_before = torch.ones_like(targets_before, dtype=torch.long)  # bsz x (1 (bos) + s)
        
        return bos_p_before_embeds, targets_before, attention_mask_before

    def _languagebind_wrap(self, input_embeds, modality):
        batch_size = input_embeds.shape[0]

        if modality == 'image':
            p_before, p_after  = '<Img>', '</Img>'
        elif modality == 'video':
            p_before, p_after  = '<Vid>', '</Vid>'
        elif modality == 'audio':
            p_before, p_after  = '<Aud>', '</Aud>'
        elif modality != 'reflector':
            raise ValueError(f'Wrong modality {modality}')

        input_embeds = input_embeds.to(self.device, dtype=self.llama_proj.weight.dtype) 
        if modality != 'reflector':
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(
                    self.device).input_ids
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(
                    self.device).input_ids
            p_before_embeds = self._llama_embeds(p_before_tokens).expand(batch_size, -1, -1)  # bsz x s1 x llama_size
            p_after_embeds = self._llama_embeds(p_after_tokens).expand(batch_size, -1, -1)  # bsz x s2 x llama_size

            llama_input_embeds = self.llama_proj(input_embeds).unsqueeze(1)  # bsz x n_tokens x llama_size
            inputs_llama = torch.cat([p_before_embeds, llama_input_embeds, p_after_embeds], dim=1)  # bsz x (s1 + n_tokens + s2) x llama_size
        else:
            inputs_llama = input_embeds  # bsz x n_tokens x llama_size
        targets_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device).fill_(-100)  # bsz x (s1 + n_tokens + s2)
        atts_llama = torch.ones_like(targets_llama, dtype=torch.long).to(self.device)  # bsz x (s1 + n_tokens + s2)
        return inputs_llama, targets_llama, atts_llama

    def _get_p_after(self, prompt, output_modality):
        """
        input:
        prompt
        ### Assistant: [IMG0]...[IMGn] [VID0]...[VIDn] [AUD0]...[AUDn]
        ###

        target:
        -100 ...
        -100 ... [IMG0]...[IMGn] [VID0]...[VIDn] [AUD0]...[AUDn]
        ###
        """
        # get tokens part by part
        batch_prompt_tokens = self.llama_tokenizer(prompt, truncation=True, max_length=self.max_length,
                                                   add_special_tokens=False).input_ids  # truncate prompts
        p_middle = '\n### Assistant:'
        p_middle_tokens = self.llama_tokenizer(p_middle, add_special_tokens=False).input_ids
        signal_tokens = ''
        output_modality = [k for k in output_modality]
        random.shuffle(output_modality)
        for m in output_modality:
            if m == 'image':
                m_signal_tokens = ' '.join([f'[IMG{i}]' for i in range(self.args['num_gen_img_tokens'])])
            elif m == 'video':
                m_signal_tokens = ' '.join([f'[VID{i}]' for i in range(self.args['num_gen_img_tokens'])])
            elif m == 'audio':
                m_signal_tokens = ' '.join([f'[AUD{i}]' for i in range(self.args['num_gen_img_tokens'])])
            else:
                raise ValueError(f'Wrong modality {m}')
            signal_tokens += ' ' + m_signal_tokens
        signal_tokens += '\n###'
        signal_token_ids = self.llama_tokenizer(signal_tokens, add_special_tokens=False).input_ids

        # concat parts and generate targets
        batch_p_after_tokens, batch_target_after = [], []
        for prompt_tokens in batch_prompt_tokens:
            p_after_tokens = prompt_tokens + p_middle_tokens
            target_after = [-100] * len(p_after_tokens)
            p_after_tokens += signal_token_ids
            target_after += signal_token_ids
            batch_p_after_tokens.append(torch.LongTensor(p_after_tokens))
            batch_target_after.append(torch.LongTensor(target_after))
        
        # pad tokens and generate attention masks
        input_ids = rnn.pad_sequence(batch_p_after_tokens, batch_first=True,
                                     padding_value=self.llama_tokenizer.pad_token_id).to(self.device)
        target_ids = rnn.pad_sequence(batch_target_after, batch_first=True, padding_value=-100).to(self.device)
        assert input_ids.size() == target_ids.size()
        attention_mask = input_ids.ne(self.llama_tokenizer.pad_token_id).long().to(self.device)
        assert attention_mask.size() == input_ids.size()
        input_embeds = self._llama_embeds(input_ids)
        return input_embeds, target_ids, attention_mask

    def prompt_wrap(self, inputs, output_modality):
        """
        :param inputs: same as train()
        :param output_modality: list[str]: keys of targets from train()

        input:
        [BOS]### Human: <Img> [Image LanguageBind] </Img> <Vid> [Video LanguageBind] </Vid> <Aud> [Audio LanguageBind] </Aud> prompt
        ### Assistant: [IMG0]...[IMGn] [VID0]...[VIDn] [AUD0]...[AUDn]
        ###

        target:
        -100 -100 ...
        -100 ... [IMG0]...[IMGn] [VID0]...[VIDn] [AUD0]...[AUDn]
        ###
        """
        assert inputs['text']
        assert len(output_modality) > 0

        batch_size = len(inputs['text'])

        p_before_embeds, targets_before, attention_mask_before = self._get_p_before(batch_size)
        p_after_embeds, targets_after, attention_mask_after = self._get_p_after(inputs['text'], output_modality)

        if len(inputs) > 1: # has multimodal inputs
            languagebind_embeds, languagebind_targets, languagebind_attention_mask = [], [], []
            input_modalities = [k for k in inputs if k != 'relector']
            random.shuffle(input_modalities)
            if self.args.get('enable_reflector') and 'relector' in inputs:
                input_modalities = ['relector'] + input_modalities
            for modality in input_modalities:
                if modality != 'text':
                    embed, target, mask = self._languagebind_wrap(inputs[modality], modality)
                    languagebind_embeds.append(embed)
                    languagebind_targets.append(target)
                    languagebind_attention_mask.append(mask)
            languagebind_embeds = torch.cat(languagebind_embeds, dim=1)  # bsz x (n_modality x s_modality) x llama_size
            languagebind_targets = torch.cat(languagebind_targets, dim=1)  # bsz x (n_modality x s_modality)
            languagebind_attention_mask = torch.cat(languagebind_attention_mask, dim=1)  # bsz x (n_modality x s_modality)

            inputs_embeds = torch.cat([p_before_embeds, languagebind_embeds, p_after_embeds], dim=1).to(self.device)  # bsz x (s1 + s2 + s3) x llama_size
            targets = torch.cat([targets_before, languagebind_targets, targets_after], dim=1).to(self.device)  # bsz x (s1 + s2 + s3)
            attention_mask = torch.cat([attention_mask_before, languagebind_attention_mask, attention_mask_after],  # bsz x (s1 + s2 + s3)
                                       dim=1).to(self.device)
        else: # prompt only
            inputs_embeds = torch.cat([p_before_embeds, p_after_embeds], dim=1).to(self.device)  # bsz x (s1 + s3) x llama_size
            targets = torch.cat([targets_before, targets_after], dim=1).to(self.device)  # bsz x (s1 + s3)
            attention_mask = torch.cat([attention_mask_before, attention_mask_after],  # bsz x (s1 + s3)
                                       dim=1).to(self.device)
                                       
        assert inputs_embeds.size()[1] == targets.size()[1]
        assert attention_mask.size() == targets.size()
        return inputs_embeds, targets, attention_mask

    def forward(self, inputs, targets, reflector_inputs=None, enable_decode=False):
        """
        :param inputs: (Dict):
            {
                'text': (List[str]): batch of action prompts (only in inputs)
                'image': (Tensor (B, 768)): LanguageBind embeddings
                'video': (Tensor (B, 768)): LanguageBind embeddings
                'audio': (Tensor (B, 768)): LanguageBind embeddings
            }
        :param targets: (Dict):
            {
                'image': (Tensor (B, 768)) / (Tensor (B, n_out_tokens, out_dim)) (if decode enabled): LanguageBind embeddings / CLIP tokens
                'video': (Tensor (B, 768)) / (Tensor (B, n_out_tokens, out_dim)) (if decode enabled): LanguageBind embeddings / CLIP tokens
                'audio': (Tensor (B, 768)) / (Tensor (B, n_out_tokens, out_dim)) (if decode enabled): LanguageBind embeddings / CLIP tokens
            }
        """
        if self.args.get('enable_reflector') and reflector_inputs:
            for k in reflector_inputs:
                reflector_inputs[k] = reflector_inputs[k].to(self.device, dtype=self.reflector.query_tokens.dtype)
            inputs['reflector'] = self.reflector(**reflector_inputs)
        inputs_embeds, llama_targets, attention_mask = self.prompt_wrap(inputs, list(targets.keys()))

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=llama_targets,
        )

        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = llama_targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)

        # based on the targets to obtain the hidden state, targets includes the [BOS] token
        hidden_states, embedding_targets = [], []
        for modality in targets:
            start_pos = (llama_targets == self.args['gen_token_idx'][modality][0]).nonzero(as_tuple=False)[:, 1].tolist()
            end_pos = (llama_targets == self.args['gen_token_idx'][modality][-1]).nonzero(as_tuple=False)[:, 1].tolist()
            assert 0 < len(start_pos) == len(end_pos) == len(inputs['text']) and len(end_pos) > 0, (start_pos, end_pos)
            hidden_embedding = []
            input_embedding = []
            for b, (s, e) in enumerate(zip(start_pos, end_pos)):
                assert e - s + 1 == self.args['num_gen_img_tokens'], (s, e)
                hidden_embedding.append(outputs.hidden_states[-1][b, s:e + 1, :])  # (n_tokens, llama_size)
                input_embedding.append(self.input_embeddings(llama_targets[b, s:e + 1]))  # (n_tokens, llama_size)
            hidden_embedding = torch.stack(hidden_embedding, dim=0)  # (B, n_tokens, llama_size)
            input_embedding = torch.stack(input_embedding, dim=0)  # (B, n_tokens, llama_size)
            hidden_states.append(self._output_proj(enable_decode, modality, hidden_embedding, input_embedding))  # (B, n_tokens, llama_size) -> (B, 1, 768) / (B, M, n_out_tokens, out_dim)
            embedding_targets.append(targets[modality].to(self.device))
        if not enable_decode:
            for embeds in hidden_states:
                embeds.squeeze_(dim=1)  # (B, 768)

        mse_loss = []
        for pred, tgt in zip(hidden_states, embedding_targets):
            mse_loss.append(l2_loss(pred, tgt).mean())
        mse_loss = torch.stack(mse_loss).mean()
        loss += self.args['mse_loss_scale'] * mse_loss
        
        return loss, gen_acc, mse_loss


    def generate_tokens_embeddings(self, input_embeds, max_tgt_len, temperature: float = 0.0, top_p: float = 1.0):
        """
        This function is used to generate the tokens and output embeddings that employed to generate images/videos/audios
        inputs: dict
        input_embeds: tensor
        return:
            out: the output tokens index
            output_embeddings: output embeddings for synthesizing images
            video_output_embedding: output embeddings for synthesizing video
            audio_output_embedding: output embeddings for synthesizing audio
        """
        max_tgt_len = min(max_tgt_len, input_embeds.shape[1] + 5)

        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=max_tgt_len,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_attentions=True
        )

        out = outputs.sequences
        output_embeddings = []
        for _hidden_states in outputs.hidden_states[1:]:
            output_embeddings.append(_hidden_states[-1])
        output_embeddings = torch.cat(output_embeddings, dim=1)

        return out, output_embeddings


    def _right_padding_to_left_padding(self, inputs, targets, attention_masks):
        batch_size = inputs.shape[0]
        new_inputs, new_targets, new_attention_masks = [], [], []
        for i in range(batch_size):
            valid_len = (attention_masks[i] == 0).nonzero(as_tuple=False)
            if valid_len.shape[0] == 0:
                valid_len = attention_masks[i].shape[0]
            else:
                valid_len = valid_len[0].item()
            new_inputs.append(torch.cat([inputs[i, valid_len:], inputs[i, :valid_len]]))
            new_targets.append(torch.cat([targets[i, valid_len:], targets[i, :valid_len]]))
            new_attention_masks.append(torch.cat([attention_masks[i, valid_len:], attention_masks[i, :valid_len]]))
        new_inputs = torch.stack(new_inputs)
        new_targets = torch.stack(new_targets)
        new_attention_masks = torch.stack(new_attention_masks)
        return new_inputs, new_targets, new_attention_masks

    def generate(self, inputs, target_modality, reflector_inputs=None, enable_decode=False,
                 max_tgt_length=256, temperature: float = 0.0, top_p: float = 1.0):
        """
            :param inputs: (Dict):
            {
                'text': (List[str]): batch of action prompts
                'image': (Tensor (B, 768)): LanguageBind embeddings
                'video': (Tensor (B, 768)): LanguageBind embeddings
                'audio': (Tensor (B, 768)): LanguageBind embeddings
            }
            :param target_modality: (List[str]): target modality names
        """
        if self.args.get('enable_reflector') and reflector_inputs:
            for k in reflector_inputs:
                reflector_inputs[k] = reflector_inputs[k].to(self.device, dtype=self.reflector.query_tokens.dtype)
            inputs['reflector'] = self.reflector(**reflector_inputs)
        inputs_embeds, llama_targets, attention_masks = self.prompt_wrap(inputs, target_modality)
        inputs_embeds, llama_targets, attention_masks = self._right_padding_to_left_padding(inputs_embeds, llama_targets, attention_masks)
        generated_ids, generated_image_embeddings = self.generate_tokens_embeddings(inputs_embeds, max_tgt_length,
                                                                                    temperature=temperature, top_p=top_p)

        # based on the targets to obtain the hidden state, targets includes the [BOS] token
        outputs = {}
        for modality in target_modality:
            start_pos = (llama_targets == self.args['gen_token_idx'][modality][0]).nonzero(as_tuple=False)[:, 1].tolist()
            end_pos = (llama_targets == self.args['gen_token_idx'][modality][-1]).nonzero(as_tuple=False)[:, 1].tolist()
            assert 0 < len(start_pos) == len(end_pos) == len(inputs['text']) and len(end_pos) > 0, (start_pos, end_pos)
            hidden_embedding = []
            input_embedding = []
            for b, (s, e) in enumerate(zip(start_pos, end_pos)):
                assert e - s + 1 == self.args['num_gen_img_tokens'], (s, e)
                # if e >= generated_image_embeddings.shape[1]:
                #     return None
                hidden_embedding.append(generated_image_embeddings[b, s:e + 1, :])  # (n_tokens, llama_size)
                input_embedding.append(self.input_embeddings(llama_targets[b, s:e + 1]))  # (n_tokens, llama_size)
            hidden_embedding = torch.stack(hidden_embedding, dim=0)  # (B, n_tokens, llama_size)
            input_embedding = torch.stack(input_embedding, dim=0)  # (B, n_tokens, llama_size)
            outputs[modality] = self._output_proj(enable_decode, modality, hidden_embedding, input_embedding)  # (B, n_tokens, llama_size) -> (B, 1, 768) / (B, n_out_tokens, out_dim)
            if not enable_decode:
                outputs[modality] = outputs[modality].squeeze(1)

        return outputs
