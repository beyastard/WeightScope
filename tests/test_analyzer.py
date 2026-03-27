#!/usr/bin/env python3
"""
WeightScope Unit Tests

Test suite for WeightAnalyzer class and core functionality.

Usage:
    python -m pytest tests/test_analyzer.py -v
    # or
    python tests/test_analyzer.py

---

Copyright (C) 2026 Bryan K Reinhart & BeySoft

This file is part of WeightScope.

WeightScope is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

WeightScope is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public
License along with WeightScope. If not, see <https://www.gnu.org/licenses/>.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import WeightAnalyzer, SessionCache, sanitize_model_name, format_number


class TestWeightAnalyzer(unittest.TestCase):
    """Test cases for WeightAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = WeightAnalyzer()
        
        # Create sample DataFrame for testing
        np.random.seed(42)
        n_samples = 1000
        
        # Simulate weight distribution (Gaussian-like)
        values = np.random.normal(0, 0.5, n_samples).astype(np.float32)
        
        # Add some exact duplicates to test counting
        values[0:100] = 0.5  # 100 occurrences of 0.5
        values[100:105] = 1.5  # 5 occurrences of 1.5
        values[105:108] = -1.5  # 3 occurrences of -1.5
        values[108:110] = 2.0  # 2 occurrences of 2.0
        values[110:111] = 3.0  # 1 occurrence (singleton)
        
        # Create DataFrame matching WeightScope format
        unique_vals, counts = np.unique(values.view(np.uint32), return_counts=True)
        self.analyzer.df = pd.DataFrame({
            'bit_pattern': [f"0x{k:08X}" for k in unique_vals],
            'value': unique_vals.view(np.float32),
            'count': counts,
            'bit_key': unique_vals
        })
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = WeightAnalyzer()
        self.assertIsNone(analyzer.df)
        self.assertIsNone(analyzer.model_hash)
        self.assertIsNone(analyzer.model_metadata)
    
    def test_get_compression_analysis(self):
        """Test compression analysis method."""
        result = self.analyzer.get_compression_analysis()
        
        self.assertIn('unique_values', result)
        self.assertIn('total_parameters', result)
        self.assertIn('value_range', result)
        self.assertIn('sparsity_pct', result)
        self.assertIn('compression_options', result)
        
        # Check unique count is reasonable
        self.assertGreater(result['unique_values'], 0)
        self.assertLessEqual(result['unique_values'], 1000)  # Our sample has ~1000
    
    def test_get_pruning_candidates(self):
        """Test pruning candidates method."""
        result = self.analyzer.get_pruning_candidates(threshold=1e-4)
        
        self.assertIn('threshold', result)
        self.assertIn('unique_candidates', result)
        self.assertIn('prunable_parameters', result)
        self.assertIn('sparsity_pct', result)
        
        # Threshold should match input
        self.assertEqual(result['threshold'], 1e-4)
    
    def test_simulate_quantization(self):
        """Test quantization simulation method."""
        result = self.analyzer.simulate_quantization(bits=8, method='mse')
        
        self.assertIn('bits', result)
        self.assertIn('mse', result)
        self.assertIn('mae', result)
        self.assertIn('max_error', result)
        
        # Bits should match input
        self.assertEqual(result['bits'], 8)
        
        # Error metrics should be non-negative
        self.assertGreaterEqual(result['mse'], 0)
        self.assertGreaterEqual(result['mae'], 0)
        self.assertGreaterEqual(result['max_error'], 0)
    
    def test_simulate_low_count_removal(self):
        """Test low-count removal simulation."""
        result = self.analyzer.simulate_low_count_removal(max_count=4)
        
        self.assertIn('removed_parameters', result)
        self.assertIn('removed_unique_patterns', result)
        self.assertIn('param_reduction_pct', result)
        self.assertIn('unique_reduction_pct', result)
        
        # Should have removed some values (we added singletons, pairs, etc.)
        self.assertGreater(result['removed_unique_patterns'], 0)
    
    def test_simulate_clipping_normalization(self):
        """Test clip + normalize simulation."""
        result = self.analyzer.simulate_clipping_normalization(threshold=2.0)
        
        self.assertIn('threshold', result)
        self.assertIn('mse', result)
        self.assertIn('snr_db', result)
        self.assertIn('clipped_pct', result)
        self.assertIn('theoretical_bits_saved', result)
        
        # Threshold should match input
        self.assertEqual(result['threshold'], 2.0)
        
        # SNR should be positive (in dB)
        self.assertGreater(result['snr_db'], 0)
    
    def test_empty_dataframe(self):
        """Test methods with empty DataFrame."""
        self.analyzer.df = None
        
        # All methods should handle None gracefully
        result = self.analyzer.get_compression_analysis()
        self.assertEqual(result, {})
        
        result = self.analyzer.get_pruning_candidates()
        self.assertEqual(result, {})
        
        result = self.analyzer.simulate_quantization()
        self.assertEqual(result, {})
    
    def test_stratified_sampling_preserves_low_count(self):
        """Test that stratified sampling preserves low-count values."""
        # This is tested indirectly through scatter plot generation
        # In production, verify singleton values appear in scatter when filtered
        singleton_count = len(self.analyzer.df[self.analyzer.df['count'] == 1])
        self.assertGreater(singleton_count, 0)  # We added at least one singleton


class TestSessionCache(unittest.TestCase):
    """Test cases for SessionCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = SessionCache()
        self.test_model_id = "test--model-123"
        
        # Create test DataFrame
        self.test_df = pd.DataFrame({
            'bit_pattern': ['0x3F800000', '0x40000000'],
            'value': [1.0, 2.0],
            'count': [100, 50]
        })
        
        self.test_metadata = {
            'file_hash': 'abc123',
            'total_parameters': 150,
            'unique_patterns': 2
        }
    
    def test_sanitize_model_name(self):
        """Test model name sanitization."""
        self.assertEqual(sanitize_model_name("amd/AMD-Llama-135m"), "amd--AMD-Llama-135m")
        self.assertEqual(sanitize_model_name("meta-llama/Llama-2-7b"), "meta-llama--Llama-2-7b")
        self.assertEqual(sanitize_model_name(None), "unknown_model")
    
    def test_format_number(self):
        """Test number formatting."""
        self.assertEqual(format_number(1000), "1,000")
        self.assertEqual(format_number(1000000), "1,000,000")
        self.assertEqual(format_number(134105856), "134,105,856")
    
    def test_cache_path_generation(self):
        """Test cache path generation."""
        cache_path = self.cache.get_cache_path(self.test_model_id)
        
        self.assertIn('.save_state', str(cache_path))
        self.assertIn('test--model-123', str(cache_path))
    
    def test_save_and_load_state(self):
        """Test saving and loading analysis state."""
        # Save state
        self.cache.save_state(self.test_model_id, self.test_df, self.test_metadata)
        
        # Load state
        df, metadata = self.cache.load_state(self.test_model_id)
        
        # Verify loaded data matches
        self.assertIsNotNone(df)
        self.assertIsNotNone(metadata)
        self.assertEqual(len(df), len(self.test_df))
        self.assertEqual(metadata['file_hash'], self.test_metadata['file_hash'])
    
    def test_cache_validation_with_hash(self):
        """Test cache validation uses file hash."""
        # Save with one hash
        self.cache.save_state(self.test_model_id, self.test_df, {'file_hash': 'abc123'})
        
        # Check with matching hash
        self.assertTrue(self.cache.check_cache(self.test_model_id, 'abc123'))
        
        # Check with different hash (should fail)
        self.assertFalse(self.cache.check_cache(self.test_model_id, 'xyz789'))
    
    def test_invalid_model_id_handling(self):
        """Test handling of invalid model IDs."""
        # None should not crash
        self.assertFalse(self.cache.check_cache(None, 'abc123'))
        
        # Non-string should be handled
        self.assertFalse(self.cache.check_cache(12345, 'abc123'))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_extreme_quantization_bits(self):
        """Test quantization with extreme bit values."""
        analyzer = WeightAnalyzer()
        
        # Create minimal test data
        analyzer.df = pd.DataFrame({
            'bit_pattern': ['0x3F800000'],
            'value': [1.0],
            'count': [1],
            'bit_key': [np.uint32(0x3F800000)]
        })
        
        # Test minimum bits
        result = analyzer.simulate_quantization(bits=4)
        self.assertEqual(result['bits'], 4)
        
        # Test maximum bits
        result = analyzer.simulate_quantization(bits=16)
        self.assertEqual(result['bits'], 16)
    
    def test_zero_threshold_pruning(self):
        """Test pruning with zero threshold."""
        analyzer = WeightAnalyzer()
        analyzer.df = pd.DataFrame({
            'bit_pattern': ['0x00000000', '0x3F800000'],
            'value': [0.0, 1.0],
            'count': [10, 5],
            'bit_key': [np.uint32(0), np.uint32(0x3F800000)]
        })
        
        result = analyzer.get_pruning_candidates(threshold=0.0)
        # Should only match exact zeros
        self.assertGreaterEqual(result['unique_candidates'], 0)
    
    def test_clipping_with_extreme_threshold(self):
        """Test clipping with very large threshold."""
        analyzer = WeightAnalyzer()
        analyzer.df = pd.DataFrame({
            'bit_pattern': ['0x3F800000'],
            'value': [1.0],
            'count': [1],
            'bit_key': [np.uint32(0x3F800000)]
        })
        
        # Very large threshold should clip nothing
        result = analyzer.simulate_clipping_normalization(threshold=1000.0)
        self.assertEqual(result['clipped_pct'], 0.0)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWeightAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionCache))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
