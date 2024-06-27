import unittest
import numpy as np
from edge.edge_utils import split_infer_clips

# Assuming split_infer_clips is defined as provided earlier

class TestSplitInferClips(unittest.TestCase):
    def setUp(self):
        # Setup basic parameters for the tests
        self.n_frames = 4
        self.J = 2
        self.D = 3

    def generate_test_data(self, frames):
        answer_1d = np.arange(frames)
        return self.correct_answer_generator(answer_1d)

    def correct_answer_generator(self, answer_1d):
        """
        [frames] -repeat-> [frames, J, D]
        """
        answer_1d = np.array(answer_1d)
        repeated = np.repeat(answer_1d, self.J * self.D)
        answer_3d = repeated.reshape(-1, self.J, self.D)
        return answer_3d

    def test_correct_answer_generator(self):
        answer_1d = np.array([1,2])
        expected_answer_3d = np.array([[[1,1,1], [1,1,1]],
                                        [[2,2,2], [2,2,2]]])
        output_answer_3d = self.correct_answer_generator(answer_1d)
        np.testing.assert_array_equal(output_answer_3d, expected_answer_3d)

    def test_discard_8(self):
        test_data = self.generate_test_data(8)
        output_data, info = split_infer_clips(test_data, self.n_frames, residual_mode='discard')
        output_sliced = output_data.reshape(-1, self.J, self.D)[info['keep_idx']]
        expected_shape = (2, 4, self.J, self.D)  # Exactly 2 full clips, no discard needed
        expected_data_3d = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7])

        self.assertEqual(output_data.shape, expected_shape)
        np.testing.assert_array_equal(output_sliced, expected_data_3d)
    def test_discard_10(self):
        test_data = self.generate_test_data(10)
        output_data, info = split_infer_clips(test_data, self.n_frames, residual_mode='discard')
        output_sliced = output_data.reshape(-1, self.J, self.D)[info['keep_idx']]
        expected_shape = (2, 4, self.J, self.D)  # Discard extra 2 frames, only 2 full clips
        expected_data_3d = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7])

        self.assertEqual(output_data.shape, expected_shape)
        np.testing.assert_array_equal(output_sliced, expected_data_3d)

    def test_fill_10(self):
        test_data = self.generate_test_data(10)
        output_data, info = split_infer_clips(test_data, self.n_frames, residual_mode='fill')
        output_sliced_filled = output_data.reshape(-1, self.J, self.D)
        output_sliced = output_sliced_filled[info['keep_idx']]
        expected_shape = (3, 4, self.J, self.D)  # Filled to make up to 12 frames total
        expected_data_3d = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_data_3d_filled = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9])

        self.assertEqual(output_data.shape, expected_shape)
        np.testing.assert_array_equal(output_sliced, expected_data_3d)  # Validate only original data is kept
        np.testing.assert_array_equal(expected_data_3d_filled, output_sliced_filled)

    def test_fill_8(self):
        test_data = self.generate_test_data(8)
        output_data, info = split_infer_clips(test_data, self.n_frames, residual_mode='fill')
        output_sliced = output_data.reshape(-1, self.J, self.D)[info['keep_idx']]
        expected_shape = (2, 4, self.J, self.D)  # Filled to make up to 8 frames, no extra needed
        expected_data_3d = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7])

        self.assertEqual(output_data.shape, expected_shape)
        np.testing.assert_array_equal(output_sliced, expected_data_3d)

    def test_backfill_10(self):
        test_data = self.generate_test_data(10)
        output_data, info = split_infer_clips(test_data, self.n_frames, residual_mode='backfill')
        output_slided_backfilled = output_data.reshape(-1, self.J, self.D)
        output_sliced = output_slided_backfilled[info['keep_idx']]
        expected_shape = (3, 4, self.J, self.D)  # Backfilled to make up to 8 frames, no extra needed
        expected_data_3d_backfilled = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7, 6, 7, 8, 9])
        expected_data_3d = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        self.assertEqual(output_data.shape, expected_shape)
        np.testing.assert_array_equal(output_sliced, expected_data_3d)  # Validate only original data is kept
        np.testing.assert_array_equal(output_slided_backfilled, expected_data_3d_backfilled)

    def test_backfill_9(self):
        test_data = self.generate_test_data(9)
        output_data, info = split_infer_clips(test_data, self.n_frames, residual_mode='backfill')
        output_slided_backfilled = output_data.reshape(-1, self.J, self.D)
        output_sliced = output_slided_backfilled[info['keep_idx']]
        expected_shape = (3, 4, self.J, self.D)  # Backfilled to make up to 8 frames, no extra needed
        expected_data_3d_backfilled = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8])
        expected_data_3d = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7, 8,])

        self.assertEqual(output_data.shape, expected_shape)
        np.testing.assert_array_equal(output_sliced, expected_data_3d)  # Validate only original data is kept
        np.testing.assert_array_equal(output_slided_backfilled, expected_data_3d_backfilled)

    def test_backfill_11(self):
        test_data = self.generate_test_data(11)
        output_data, info = split_infer_clips(test_data, self.n_frames, residual_mode='backfill')
        output_slided_backfilled = output_data.reshape(-1, self.J, self.D)
        output_sliced = output_slided_backfilled[info['keep_idx']]
        expected_shape = (3, 4, self.J, self.D)  # Backfilled to make up to 8 frames, no extra needed
        expected_data_3d_backfilled = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10])
        expected_data_3d = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        self.assertEqual(output_data.shape, expected_shape)
        np.testing.assert_array_equal(output_sliced, expected_data_3d)  # Validate only original data is kept
        np.testing.assert_array_equal(output_slided_backfilled, expected_data_3d_backfilled)

    def test_backfill_8(self):
        test_data = self.generate_test_data(8)
        output_data, info = split_infer_clips(test_data, self.n_frames, residual_mode='backfill')
        output_slided_backfilled = output_data.reshape(-1, self.J, self.D)
        output_sliced = output_slided_backfilled[info['keep_idx']]
        expected_shape = (2, 4, self.J, self.D)  # Backfilled to make up to 8 frames, no extra needed
        expected_data_3d_backfilled = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7])
        expected_data_3d = self.correct_answer_generator([0, 1, 2, 3, 4, 5, 6, 7])

        self.assertEqual(output_data.shape, expected_shape)
        np.testing.assert_array_equal(output_sliced, expected_data_3d)  # Validate only original data is kept
        np.testing.assert_array_equal(output_slided_backfilled, expected_data_3d_backfilled)

if __name__ == '__main__':
    unittest.main()