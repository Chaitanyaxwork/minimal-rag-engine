import unittest
from chunker import chunk_text, split_sentences, split_long_sentence


class TestChunker(unittest.TestCase):
    def test_empty_and_whitespace(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("  \n\t  "), [])

    def test_short_document_is_one_chunk(self):
        self.assertEqual(chunk_text("Hello world. Bye."), ["Hello world. Bye."])

    def test_worked_example(self):
        text = "Cats purr. Dogs bark. Birds sing. Fish swim."
        self.assertEqual(
            chunk_text(text, chunk_size=25, overlap=11),
            ["Cats purr. Dogs bark.", "Dogs bark. Birds sing.", "Birds sing. Fish swim."],
        )

    def test_overlap_falls_back_to_word_tail(self):
        # "Birds sing." is 11 chars > overlap 10, so only a word-aligned tail is repeated
        text = "Cats purr. Dogs bark. Birds sing. Fish swim."
        self.assertEqual(chunk_text(text, 25, 10)[-1], "sing. Fish swim.")

    def test_size_limit_and_no_empty_chunks(self):
        text = ("The quick brown fox jumps over the lazy dog. " * 40) + "x" * 700
        for size, ov in [(50, 10), (120, 30), (500, 100), (10, 0), (7, 6)]:
            chunks = chunk_text(text, size, ov)
            self.assertTrue(chunks)
            for c in chunks:
                self.assertTrue(c.strip())
                self.assertLessEqual(len(c), size)

    def test_long_sentence_splits_on_words_with_overlap(self):
        text = " ".join(f"word{i}" for i in range(60))  # one sentence, no punctuation
        chunks = chunk_text(text, chunk_size=60, overlap=15)
        self.assertGreater(len(chunks), 1)
        for a, b in zip(chunks, chunks[1:]):
            self.assertIn(b.split()[0], a.split())    # next chunk starts with repeated word

    def test_no_space_hard_cut(self):
        pieces = split_long_sentence("a" * 25, 10)
        self.assertEqual(pieces, ["a" * 10, "a" * 10, "a" * 5])

    def test_nothing_lost(self):
        text = "One fish. Two fish! Red fish? Blue fish."
        joined = " ".join(chunk_text(text, 20, 0))
        self.assertEqual(joined, text)

    def test_punctuation_edge_cases(self):
        self.assertEqual(split_sentences('He said "stop." Then left.'),
                         ['He said "stop."', "Then left."])
        self.assertEqual(split_sentences("Price is 3.14 dollars."), ["Price is 3.14 dollars."])

    def test_validation(self):
        with self.assertRaises(ValueError): chunk_text("x", 0, 0)
        with self.assertRaises(ValueError): chunk_text("x", 10, -1)
        with self.assertRaises(ValueError): chunk_text("x", 10, 10)
        with self.assertRaises(TypeError): chunk_text(None)


if __name__ == "__main__":
    unittest.main()
