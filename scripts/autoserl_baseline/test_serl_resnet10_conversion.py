"""Deterministic weight-map and geometry contract checks, no model download.

This is a unit regression for the official JAX->PyTorch representation, not
an assertion that original weights or Ascend forward parity were verified.
"""
import unittest

import numpy as np
import torch
from torch import nn

from acoserl.serl_resnet10 import (
    FEATURE_CHANNELS, FEATURE_SIDE, SERLResNet10,
    official_weights_from_jax, same_pad,
)


def synthetic_flax_tree(model):
    """Inverse-map a freshly initialized Torch network to Flax weight names."""
    state = model.state_dict()
    conv = lambda key: {'kernel': state[key].detach().numpy().transpose(2,3,1,0).copy()}
    norm = lambda key: {'scale': state[key+'.weight'].detach().numpy().copy(),
                        'bias': state[key+'.bias'].detach().numpy().copy()}
    result = {'conv_init':conv('conv_init.weight'),
              'norm_init':norm('norm_init'),
              'output_head':{
                  'kernel': state['output_head.weight'].detach().numpy().T.copy(),
                  'bias': state['output_head.bias'].detach().numpy().copy(),
              }}
    for index in range(4):
        base = f'blocks.{index}'
        block = {
            'Conv_0':conv(base+'.conv1.weight'),
            'Conv_1':conv(base+'.conv2.weight'),
            'MyGroupNorm_0':norm(base+'.norm1'),
            'MyGroupNorm_1':norm(base+'.norm2'),
        }
        if index:
            block['conv_proj'] = conv(base+'.shortcut.0.weight')
            block['norm_proj'] = norm(base+'.shortcut.1')
        result[f'ResNetBlock_{index}'] = block
    return result


class SERLResNet10ConversionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1234)
        torch.set_num_threads(2)
        cls.model = SERLResNet10().eval()
        cls.synthetic_tree = synthetic_flax_tree(cls.model)

    def test_all_four_official_resnet_blocks_map_to_exact_tensor_values(self):
        restored = official_weights_from_jax(self.synthetic_tree)
        self.assertEqual(set(restored), set(self.model.state_dict()))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, restored[key]), key)

    def test_invalid_or_partial_official_checkpoint_is_rejected(self):
        wrong = dict(self.synthetic_tree)
        wrong.pop('ResNetBlock_3')
        with self.assertRaisesRegex(ValueError, 'top-level keys mismatch'):
            official_weights_from_jax(wrong)

    def test_missing_official_output_head_is_rejected(self):
        wrong = dict(self.synthetic_tree)
        wrong.pop('output_head')
        with self.assertRaisesRegex(ValueError, 'output_head'):
            official_weights_from_jax(wrong)

    def test_unrecognized_top_level_parameter_is_rejected(self):
        wrong = dict(self.synthetic_tree)
        wrong['arbitrary_extra'] = {'kernel': np.zeros((1,), dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, 'arbitrary_extra'):
            official_weights_from_jax(wrong)

    def test_output_head_bad_nested_keys_are_rejected(self):
        wrong = dict(self.synthetic_tree)
        wrong['output_head'] = {'kernel': self.synthetic_tree['output_head']['kernel']}
        with self.assertRaisesRegex(ValueError, 'output_head keys mismatch'):
            official_weights_from_jax(wrong)

    def test_output_head_wrong_class_count_is_rejected(self):
        wrong = dict(self.synthetic_tree)
        wrong['output_head'] = {
            'kernel': np.zeros((512, 999), dtype=np.float32),
            'bias': np.zeros((999,), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, 'output_head shape mismatch'):
            official_weights_from_jax(wrong)

    def test_output_head_nonfinite_weights_are_rejected(self):
        wrong = dict(self.synthetic_tree)
        broken = dict(wrong['output_head'])
        broken['bias'] = broken['bias'].copy()
        broken['bias'][0] = np.nan
        wrong['output_head'] = broken
        with self.assertRaisesRegex(ValueError, 'output_head must have finite'):
            official_weights_from_jax(wrong)

    def test_classifier_head_is_preserved_with_flax_dense_transpose(self):
        mapped = official_weights_from_jax(self.synthetic_tree)
        self.assertEqual(tuple(mapped['output_head.weight'].shape), (1000, 512))
        self.assertEqual(tuple(mapped['output_head.bias'].shape), (1000,))
        expected = torch.from_numpy(
            self.synthetic_tree['output_head']['kernel'].T.copy())
        self.assertTrue(torch.equal(mapped['output_head.weight'], expected))
        self.assertTrue(torch.equal(
            mapped['output_head.bias'],
            torch.from_numpy(self.synthetic_tree['output_head']['bias'].copy())))

    def test_shape_mismatch_is_rejected(self):
        wrong = dict(self.synthetic_tree)
        wrong['conv_init'] = {'kernel':np.zeros((7,7,3,32), dtype=np.float32)}
        with self.assertRaises(ValueError):
            official_weights_from_jax(wrong)

    def test_groupnorm_four_groups_not_batchnorm(self):
        self.assertTrue(all(isinstance(m, nn.GroupNorm) and m.num_groups == 4
                            for m in self.model.modules() if isinstance(m, nn.GroupNorm)))
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d)
                             for m in self.model.modules()))

    def test_flax_same_padding_is_asymmetric_on_even_strided_image(self):
        x = torch.arange(32*32,dtype=torch.float32).reshape(1,1,32,32)
        padded = same_pad(x,3,2)
        self.assertEqual(padded.shape, (1,1,33,33))
        self.assertTrue(torch.equal(padded[:,:,:32,:32], x))
        self.assertEqual(padded[:,:,-1,:].abs().sum().item(), 0.)

    def test_stage2_spatial_and_complete_four_stage_dimensions(self):
        with torch.no_grad():
            x = torch.rand(1,3,84,84)
            spatial = self.model(x,spatial=True)
            full = self.model(x,spatial=False)
        self.assertEqual(spatial.shape,(1,FEATURE_CHANNELS,FEATURE_SIDE,FEATURE_SIDE))
        self.assertEqual(full.shape,(1,512,4,4))
        self.assertTrue(torch.isfinite(spatial).all())

    def test_frozen_backbone_has_no_backward_but_adapter_learns(self):
        from acoserl.serl_resnet10 import FEATURE_CHANNELS
        frozen = SERLResNet10().eval().requires_grad_(False)
        adapter = nn.Linear(FEATURE_CHANNELS,128)
        with torch.no_grad():
            features = frozen(torch.rand(1,3,84,84))
        loss = adapter(features.flatten(2).transpose(1,2)).square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))


if __name__ == '__main__':
    unittest.main()
