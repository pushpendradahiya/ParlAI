"""
Transformer Agents.
"""
from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import recursive_getattr
from parlai.utils.logging import logging

from parlai.agents.transformer.modules import TransformerGeneratorModel

import torch


from itertools import chain
from functools import lru_cache

import torch as th
import numpy as np

from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs

from parlai.agents.transformer.transformer import TransformerGeneratorAgent

from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE

TOKEN_DIALOG = '__dialog__'


DEFAULT_OPTS = {
    "learningrate": 5e-4,
    "optimizer": "adam",
    "lr_scheduler": "invsqrt",
    "warmup_updates": 5000,
    "clip_norm": 0.1,
    "ffn_size": 512,
    "embedding_size": 256,
    "n_heads": 2,
    "dropout": 0.2,
    "n_layers": 5,
    "betas": "0.9,0.98",
    "truncate": 128,
    "add_token_knowledge": True,
    "dict_textfields": "text,labels,chosen_topic,checked_sentence,knowledge,title",
}


def add_common_cmdline_args(argparser):
    """
    Add common command line args.
    """
    argparser.add_argument(
        '-esz',
        '--embedding-size',
        type=int,
        default=300,
        help='Size of all embedding layers',
    )
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument(
        '-hid',
        '--ffn-size',
        type=int,
        default=300,
        help='Hidden size of the FFN layers',
    )
    argparser.add_argument(
        '--dropout', type=float, default=0.0, help='Dropout used in Vaswani 2017.'
    )
    argparser.add_argument(
        '--attention-dropout',
        type=float,
        default=0.0,
        help='Dropout used after attention softmax.',
    )
    argparser.add_argument(
        '--relu-dropout',
        type=float,
        default=0.0,
        help='Dropout used after ReLU. From tensor2tensor.',
    )
    argparser.add_argument(
        '--n-heads', type=int, default=2, help='Number of multihead attention heads'
    )
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)
    argparser.add_argument(
        '--n-positions',
        type=int,
        default=None,
        hidden=True,
        help='Number of positional embeddings to learn. Defaults '
        'to truncate or 1024 if not provided.',
    )
    argparser.add_argument(
        '--n-segments',
        type=int,
        default=0,
        help='The number of segments that support the model. '
        'If zero no segment and no langs_embedding.',
    )
    argparser.add_argument(
        '--variant',
        choices={'aiayn', 'xlm', 'prelayernorm', 'bart'},
        default='aiayn',
        help='Chooses locations of layer norms, etc. prelayernorm '
        'is used to match some fairseq models',
        recommended='xlm',
    )
    argparser.add_argument(
        '--activation',
        choices={'relu', 'gelu'},
        default='relu',
        help='Nonlinear activation to use. AIAYN uses relu, but '
        'more recent papers prefer gelu.',
        recommended='gelu',
    )
    argparser.add_argument(
        '--output-scaling',
        type=float,
        default=1.0,
        help='scale the output of every transformer by this quantity.',
    )
    argparser.add_argument(
        '--share-word-embeddings',
        type='bool',
        default=True,
        help='Share word embeddings table for candidate and context'
        'in the memory network',
    )
    argparser.add_argument(
        '-nel',
        '--n-encoder-layers',
        type=int,
        default=-1,
        help='This will overide the n-layers for asymmetrical transformers',
    )
    argparser.add_argument(
        '-ndl',
        '--n-decoder-layers',
        type=int,
        default=-1,
        help='This will overide the n-layers for asymmetrical transformers',
    )
    argparser.add_argument(
        '--model-parallel',
        type='bool',
        default=False,
        help='Shard the layers across multiple GPUs.',
    )

class _GenericWizardAgent(TransformerGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.set_defaults(**DEFAULT_OPTS)
        super(_GenericWizardAgent, cls).add_cmdline_args(argparser)

    def batchify(self, obs_batch):
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]

        checked_sentences = []
        for obs in reordered_observations:
            checked_sentence = '{} {} {}'.format(
                obs.get('title', ''), TOKEN_KNOWLEDGE, obs.get('checked_sentence', '')
            )
            checked_sentences.append(checked_sentence)

        batch['checked_sentence'] = checked_sentences

        return batch


class TransformerRetGenAgent(_GenericWizardAgent):
    """
    TransformerGeneratorAgent.

    Implementation of TorchGeneratorAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Transformer Arguments')
        agent.add_argument(
            '--knowledge-alpha',
            type=float,
            default=0.95,
            help='Weight on the knowledge-attn loss',
        )
        agent.add_argument(
            '--knowledge-truncate',
            type=int,
            default=32,
            help='Knowledge truncation field. Defaults to same as --truncate.',
        )
        agent.add_argument(
            '--max-knowledge',
            type=int,
            help='Reduce the amount of negative knowledge at train time.',
        )
        argparser.add_argument(
            '--knowledge-alpha',
            type=float,
            default=0.95,
            help='Weight on the knowledge-attn loss',
        )
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(TransformerRetGenAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def _resize_token_embeddings(self, state_dict, msg=None):
        """
        Resize the token embeddings when are adding extra special tokens.
        """
        # map extra special tokens carefully
        new_size = self.model.embeddings.weight.size()[0]
        orig_size = state_dict['embeddings.weight'].size()[0]
        logging.info(f'Resizing token embeddings from {orig_size} to {new_size}')
        if new_size <= orig_size:
            # new size should be greater than original size,
            # as we are adding special tokens
            raise RuntimeError(msg)

        for emb_weights in [
            'embeddings.weight',
            'encoder.embeddings.weight',
            'decoder.embeddings.weight',
        ]:
            # get new_embs
            old_embs = state_dict[emb_weights]
            new_embs = recursive_getattr(self.model, emb_weights).to(old_embs.device)
            # copy over old weights
            new_embs.data[:orig_size, :] = old_embs.data[:orig_size, :]
            # reset in state dict
            state_dict[emb_weights] = new_embs

        return state_dict

    def _set_text_vec(self, obs, history, truncate):
        """
        Set the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        # print("\n\n Obs: ", obs)
        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs

            obs['full_text'] = history_string
            # print("\n\n Checked vec: ", checked_vec)
            if history_string:
                obs['text_vec'] = history.get_history_vec()
                obs['full_text_vec'] = history.get_history_vec()

            # if wizard of wikipedia, add the retreived text.
            if obs['id'] == 'wizard_of_wikipedia':
                checked_sentence = '{} {}'.format(TOKEN_KNOWLEDGE, obs['checked_sentence']
                                                  )
                obs.force_set('full_text', history_string + checked_sentence)
                checked_vec = self._vectorize_text(
                    # the beginning of the sentence is more useful
                    checked_sentence,
                    truncate=self.knowledge_truncate,
                    add_end=True,
                    truncate_left=False,
                )
                obs['text_vec'].extend(checked_vec.tolist())
                obs['full_text_vec'].extend(checked_vec.tolist())
                # print("\n\n text vec: ", obs['full_text'], obs['text_vec'], history.get_history_vec().extend(checked_vec))
        # check truncation
        if obs.get('text_vec') is not None:
            truncate_left = not self.history_reversed
            truncated_vec = self._check_truncate(
                obs['text_vec'], truncate, truncate_left
            )
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))

        return obs


    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self._vectorize_text = lru_cache(int(2 ** 20))(self._vectorize_text)

        # knowledge truncate defaults to the same as --truncate
        self.knowledge_truncate = opt.get('knowledge_truncate')
        if not self.knowledge_truncate:
            self.knowledge_truncate = opt['truncate']
        self.max_knowledge = opt.get('max_knowledge')
        self.knowledge_alpha = opt['knowledge_alpha']
    #
    # def _dummy_batch(self, bsz, maxlen):
    #     batch = super()._dummy_batch(bsz, maxlen)
    #     batch['know_vec'] = th.zeros(bsz, 2, 2).long().cuda()
    #     # bool/uint8 backwards for pytorch 1.0/1.2 compatibility
    #     ck_mask = (th.ones(bsz, 2, dtype=th.uint8) != 0).cuda()
    #     batch['ck_mask'] = ck_mask
    #     batch['cs_ids'] = th.zeros(bsz).long().cuda()
    #     batch['use_cs_ids'] = True
    #     return batch
    #
    # def compute_loss(self, batch, return_output=False):
    #     # first compute our regular forced decoding loss
    #     token_loss, model_output = super().compute_loss(batch, return_output=True)
    #     notnull = batch.label_vec.ne(self.NULL_IDX)
    #     num_tokens = notnull.long().sum().item()
    #
    #     encoder_states = model_output[2]
    #     ctx_know_attn = encoder_states[2]
    #
    #     if self.knowledge_alpha == 0.0:
    #         loss = token_loss
    #     else:
    #         _, know_pred = ctx_know_attn.max(1)
    #         know_acc = (know_pred == batch.cs_ids).float().sum().item()
    #         know_chance = batch.ck_mask.sum(1).float().reciprocal().sum().item()
    #         self.metrics['know_chance'] += know_chance
    #         self.metrics['bsz'] += batch.text_vec.size(0)
    #         self.metrics['know_acc'] += know_acc
    #         know_loss = th.nn.functional.cross_entropy(
    #             ctx_know_attn, batch.cs_ids, reduction='mean'
    #         )
    #         self.metrics['know_loss'] += know_loss.item() * batch.text_vec.size(0)
    #         # in the original paper the loss was scaled by num_tokens for both
    #         # know_loss and token_loss
    #         know_loss /= num_tokens
    #         loss = (
    #             1 - self.knowledge_alpha
    #         ) * token_loss + self.knowledge_alpha * know_loss
    #     if return_output:
    #         return (loss, model_output)
    #     else:
    #         return loss
    #
    # def reset_metrics(self):
    #     super().reset_metrics()
    #     self.metrics['bsz'] = 0.0
    #     self.metrics['know_acc'] = 0.0
    #     self.metrics['know_loss'] = 0.0
    #     self.metrics['know_chance'] = 0.0
    #
    # def report(self):
    #     r = super().report()
    #     bsz = max(self.metrics['bsz'], 1)
    #     for k in ['know_loss', 'know_acc', 'know_chance']:
    #         # round and average across all items since last report
    #         r[k] = round_sigfigs(self.metrics[k] / bsz, 4)
    #     return r
    #
    # def _parse_knowledge(self, obs):
    #     if 'knowledge_parsed' in obs:
    #         # make a copy of the list to prevent the future padding step from
    #         # being destructive
    #         return list(obs['knowledge_parsed'])
    #
    #     if 'checked_sentence' not in obs:
    #         # interactive time. we're totally on our own
    #         obs_know = [
    #             k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')
    #         ]
    #         obs_know = [k for k in obs_know if k]
    #         obs['knowledge_parsed'] = obs_know
    #         # print("obs_know: ", obs_know)
    #         return obs['knowledge_parsed']
    #
    #     checked_sentence = '{} {} {}'.format(
    #         obs['title'], TOKEN_KNOWLEDGE, obs['checked_sentence']
    #     )
    #     # grab all the nonempty knowledge
    #     obs_know = [
    #         k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')
    #     ]
    #     obs_know = [k for k in obs_know if k]
    #
    #     # we want the correct knowledge to always be in index 0
    #     try:
    #         i = obs_know.index(checked_sentence)
    #     except ValueError:
    #         # uh oh, couldn't find the sentence in the knowledge. This happens for
    #         # one or two examples in the training set. We can just artificially
    #         # put it back in
    #         i = 0
    #         obs_know[0] = checked_sentence
    #     obs_know[0], obs_know[i] = obs_know[i], obs_know[0]
    #
    #     obs['knowledge_parsed'] = obs_know
    #     obs['checked_sentence_parsed'] = checked_sentence
    #     # print("checked_sentence: ", checked_sentence)
    #     return obs['knowledge_parsed']
    #
    # def batchify(self, obs_batch):
    #     """
    #     Wizard custom batchify, which passes along the knowledge/title.
    #
    #     Following the docstring of TorchAgent.batchify, it calls super, then
    #     uses an extended version of the torch_agent.Batch namedtuple.
    #
    #     The purpose of extending the info is to keep track of some custom
    #     metrics.
    #     """
    #     batch = super().batchify(obs_batch)
    #     reordered_observations = [obs_batch[i] for i in batch.valid_indices]
    #     is_training = 'labels' in reordered_observations[0]
    #
    #     # first parse and compile all the knowledge together
    #     all_knowledges = []  # list-of-lists knowledge items for each observation
    #     knowledge_counts = []  # how much knowledge each observation gets
    #     for obs in reordered_observations:
    #         obs_know = self._parse_knowledge(obs)
    #         # downsample if desired
    #         if (
    #             is_training
    #             and self.max_knowledge
    #             and len(obs_know) > self.max_knowledge
    #         ):
    #             # offset by one so that we don't choose 0
    #             keepers = 1 + np.random.choice(
    #                 len(obs_know) - 1, self.max_knowledge, False
    #             )
    #             # correct answer is always the first one
    #             keepers[0] = 0
    #             obs_know = [obs_know[i] for i in keepers]
    #         all_knowledges.append(obs_know)
    #         knowledge_counts.append(len(obs_know))
    #
    #     # now we want to actually pack this into a tensor, along with the mask
    #     N = len(reordered_observations)
    #     K = max(knowledge_counts)
    #     # round out the array so everything is equally sized
    #     for i in range(N):
    #         all_knowledges[i] += [''] * (K - knowledge_counts[i])
    #     flattened_knowledge = list(chain(*all_knowledges))
    #
    #     knowledge_vec = [
    #         self._vectorize_text(
    #             # the beginning of the sentence is more useful
    #             k,
    #             truncate=self.knowledge_truncate,
    #             add_end=True,
    #             truncate_left=False,
    #         )
    #         for k in flattened_knowledge
    #     ]
    #     knowledge_vec, _ = padded_tensor(
    #         knowledge_vec, self.NULL_IDX, self.use_cuda, left_padded=True
    #     )
    #     knowledge_vec[:, -1] = self.END_IDX
    #     T = knowledge_vec.size(-1)
    #     knowledge_vec = knowledge_vec.view(N, K, T)
    #
    #     # knowledge mask is a N x K tensor saying which items we're allowed to
    #     # attend over
    #     bsz = len(reordered_observations)
    #     ck_mask = th.zeros(bsz, K, dtype=th.uint8)
    #     for i, klen in enumerate(knowledge_counts):
    #         ck_mask[i, :klen] = 1
    #     ck_mask = ck_mask != 0  # for pytorch 1.0/1.2 uint8/bool compatibility
    #     # and the correct labels
    #     cs_ids = th.LongTensor(bsz).zero_()
    #
    #     if self.use_cuda:
    #         knowledge_vec = knowledge_vec.cuda()
    #         ck_mask = ck_mask.cuda()
    #         cs_ids = cs_ids.cuda()
    #
    #     batch['know_vec'] = knowledge_vec
    #     batch['ck_mask'] = ck_mask
    #     batch['cs_ids'] = cs_ids
    #     batch['use_cs_ids'] = is_training
    #     batch['knowledge'] = np.array(flattened_knowledge).reshape(N, K)
    #     print("batch: ", batch)
    #     return batch
    #
    #
    # def _model_input(self, batch):
    #     return (
    #         batch.text_vec,
    #         batch.know_vec,
    #         batch.ck_mask,
    #         batch.cs_ids,
    #         batch.use_cs_ids,
    #     )






class EndToEndAgent(_GenericWizardAgent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self._vectorize_text = lru_cache(int(2 ** 20))(self._vectorize_text)

        # knowledge truncate defaults to the same as --truncate
        self.knowledge_truncate = opt.get('knowledge_truncate')
        if not self.knowledge_truncate:
            self.knowledge_truncate = opt['truncate']
        self.max_knowledge = opt.get('max_knowledge')
        self.knowledge_alpha = opt['knowledge_alpha']

    def _dummy_batch(self, bsz, maxlen):
        batch = super()._dummy_batch(bsz, maxlen)
        batch['know_vec'] = th.zeros(bsz, 2, 2).long().cuda()
        # bool/uint8 backwards for pytorch 1.0/1.2 compatibility
        ck_mask = (th.ones(bsz, 2, dtype=th.uint8) != 0).cuda()
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = th.zeros(bsz).long().cuda()
        batch['use_cs_ids'] = True
        return batch

    def compute_loss(self, batch, return_output=False):
        # first compute our regular forced decoding loss
        token_loss, model_output = super().compute_loss(batch, return_output=True)
        notnull = batch.label_vec.ne(self.NULL_IDX)
        num_tokens = notnull.long().sum().item()

        encoder_states = model_output[2]
        ctx_know_attn = encoder_states[2]

        if self.knowledge_alpha == 0.0:
            loss = token_loss
        else:
            _, know_pred = ctx_know_attn.max(1)
            know_acc = (know_pred == batch.cs_ids).float().sum().item()
            know_chance = batch.ck_mask.sum(1).float().reciprocal().sum().item()
            self.metrics['know_chance'] += know_chance
            self.metrics['bsz'] += batch.text_vec.size(0)
            self.metrics['know_acc'] += know_acc
            know_loss = th.nn.functional.cross_entropy(
                ctx_know_attn, batch.cs_ids, reduction='mean'
            )
            self.metrics['know_loss'] += know_loss.item() * batch.text_vec.size(0)
            # in the original paper the loss was scaled by num_tokens for both
            # know_loss and token_loss
            know_loss /= num_tokens
            loss = (
                1 - self.knowledge_alpha
            ) * token_loss + self.knowledge_alpha * know_loss
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['bsz'] = 0.0
        self.metrics['know_acc'] = 0.0
        self.metrics['know_loss'] = 0.0
        self.metrics['know_chance'] = 0.0

    def report(self):
        r = super().report()
        bsz = max(self.metrics['bsz'], 1)
        for k in ['know_loss', 'know_acc', 'know_chance']:
            # round and average across all items since last report
            r[k] = round_sigfigs(self.metrics[k] / bsz, 4)
        return r

    def _parse_knowledge(self, obs):
        if 'knowledge_parsed' in obs:
            # make a copy of the list to prevent the future padding step from
            # being destructive
            return list(obs['knowledge_parsed'])

        if 'checked_sentence' not in obs:
            # interactive time. we're totally on our own
            obs_know = [
                k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')
            ]
            obs_know = [k for k in obs_know if k]
            obs['knowledge_parsed'] = obs_know
            # print("obs_know: ", obs_know)
            return obs['knowledge_parsed']

        checked_sentence = '{} {} {}'.format(
            obs['title'], TOKEN_KNOWLEDGE, obs['checked_sentence']
        )
        # grab all the nonempty knowledge
        obs_know = [
            k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')
        ]
        obs_know = [k for k in obs_know if k]

        # we want the correct knowledge to always be in index 0
        try:
            i = obs_know.index(checked_sentence)
        except ValueError:
            # uh oh, couldn't find the sentence in the knowledge. This happens for
            # one or two examples in the training set. We can just artificially
            # put it back in
            i = 0
            obs_know[0] = checked_sentence
        obs_know[0], obs_know[i] = obs_know[i], obs_know[0]

        obs['knowledge_parsed'] = obs_know
        obs['checked_sentence_parsed'] = checked_sentence
        # print("checked_sentence: ", checked_sentence)
        return obs['knowledge_parsed']

    def batchify(self, obs_batch):
        """
        Wizard custom batchify, which passes along the knowledge/title.

        Following the docstring of TorchAgent.batchify, it calls super, then
        uses an extended version of the torch_agent.Batch namedtuple.

        The purpose of extending the info is to keep track of some custom
        metrics.
        """
        batch = super().batchify(obs_batch)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]
        is_training = 'labels' in reordered_observations[0]

        # first parse and compile all the knowledge together
        all_knowledges = []  # list-of-lists knowledge items for each observation
        knowledge_counts = []  # how much knowledge each observation gets
        for obs in reordered_observations:
            obs_know = self._parse_knowledge(obs)
            # downsample if desired
            if (
                is_training
                and self.max_knowledge
                and len(obs_know) > self.max_knowledge
            ):
                # offset by one so that we don't choose 0
                keepers = 1 + np.random.choice(
                    len(obs_know) - 1, self.max_knowledge, False
                )
                # correct answer is always the first one
                keepers[0] = 0
                obs_know = [obs_know[i] for i in keepers]
            all_knowledges.append(obs_know)
            knowledge_counts.append(len(obs_know))

        # now we want to actually pack this into a tensor, along with the mask
        N = len(reordered_observations)
        K = max(knowledge_counts)
        # round out the array so everything is equally sized
        for i in range(N):
            all_knowledges[i] += [''] * (K - knowledge_counts[i])
        flattened_knowledge = list(chain(*all_knowledges))

        knowledge_vec = [
            self._vectorize_text(
                # the beginning of the sentence is more useful
                k,
                truncate=self.knowledge_truncate,
                add_end=True,
                truncate_left=False,
            )
            for k in flattened_knowledge
        ]
        knowledge_vec, _ = padded_tensor(
            knowledge_vec, self.NULL_IDX, self.use_cuda, left_padded=True
        )
        knowledge_vec[:, -1] = self.END_IDX
        T = knowledge_vec.size(-1)
        knowledge_vec = knowledge_vec.view(N, K, T)

        # knowledge mask is a N x K tensor saying which items we're allowed to
        # attend over
        bsz = len(reordered_observations)
        ck_mask = th.zeros(bsz, K, dtype=th.uint8)
        for i, klen in enumerate(knowledge_counts):
            ck_mask[i, :klen] = 1
        ck_mask = ck_mask != 0  # for pytorch 1.0/1.2 uint8/bool compatibility
        # and the correct labels
        cs_ids = th.LongTensor(bsz).zero_()

        if self.use_cuda:
            knowledge_vec = knowledge_vec.cuda()
            ck_mask = ck_mask.cuda()
            cs_ids = cs_ids.cuda()

        batch['know_vec'] = knowledge_vec
        batch['ck_mask'] = ck_mask
        batch['cs_ids'] = cs_ids
        batch['use_cs_ids'] = is_training
        batch['knowledge'] = np.array(flattened_knowledge).reshape(N, K)
        # print("batch: ", batch)
        return batch

    @classmethod
    def add_cmdline_args(cls, argparser):
        super(EndToEndAgent, cls).add_cmdline_args(argparser)
        group = argparser.add_argument_group("EndToEnd Agent")
        group.add_argument(
            '--knowledge-alpha',
            type=float,
            default=0.95,
            help='Weight on the knowledge-attn loss',
        )
        group.add_argument(
            '--knowledge-truncate',
            type=int,
            default=32,
            help='Knowledge truncation field. Defaults to same as --truncate.',
        )
        group.add_argument(
            '--max-knowledge',
            type=int,
            help='Reduce the amount of negative knowledge at train time.',
        )
        argparser.add_argument(
            '--knowledge-alpha',
            type=float,
            default=0.95,
            help='Weight on the knowledge-attn loss',
        )

    def _model_input(self, batch):
        return (
            batch.text_vec,
            batch.know_vec,
            batch.ck_mask,
            batch.cs_ids,
            batch.use_cs_ids,
        )

    def build_model(self):
        self.model = EndToEndModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model = self.model.cuda()
        return self.model
