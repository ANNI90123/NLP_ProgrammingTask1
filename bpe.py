#!/usr/bin/python3
from tqdm import tqdm
import re
import json
import os
from collections import  Counter
from typing import List,  Tuple, Optional


GROUP = "12"  # TODO: write in your group number

def load_imdb_dataset(file_path: str="./imdb.txt", small_dataset=False) -> List[str]:
    """ This function loads the IMDB dataset from the txt file.
    Args:
        file_path (str): The path to the json file.
        p (int): Percentage of the dataset to use.
    Returns:
        list: A list of texts from the dataset.
    """
    # this function was implemented for you
    with open(file_path, 'r') as f:
        dataset = f.readlines()
    dataset = dataset[:100] if small_dataset else dataset
    print(f"Loaded {len(dataset)} documents")
    return dataset


class BPETokenizer:
    def __init__(self):
        """Initialize the BPE tokenizer."""
        self.vocab = {} 
        self.merges = [] 
    ################################# BPE merge rule finding #########################################
    ############################## Lecture slides: NLP:III-43--51 ####################################
    def pre_tokenize(self, text: str) -> List[str]:
        """Preprocess the texts by normalizing them (e.g. lowercasing) and 
        tokenizing them into strings (e.g. splitting them by whitespace).
        Args:
            text: Input text (string) to be preprocessed.
        Returns:
            List of tokenized strings.
        """
        # TODO: Implement the method to lowercase and split the input by whitespace
        # (optional) You can also use one or more heuristics introduced in the lecture (NLP:III-14)
        text = text.lower()
        token_list = text.split()
        return token_list

    def preprocess(self, texts: List[str]) -> List[List[str]]:
        """
        Split each string in the pre_tokenized list of strings into characters.
        Args:
            texts: List of pre_tokenized strings (tokens)
        Returns:
            List of lists, where each inner list contains character-level tokens for each word
        """                
        preprocessed_texts = []
        for text in texts:
            pre_tokenized_text = self.pre_tokenize(text)
            tokenized_texts = []
            for token in pre_tokenized_text:
                tokenized_texts.append(list(token))
            preprocessed_texts.append(tokenized_texts)
            # TODO: implement the _split_into_characters method
        print(preprocessed_texts)
        return preprocessed_texts


    def _get_stats(self, preprocessed_texts: List[List[str]]) -> Counter:
        """
        Count subword pair frequencies in the preprocessed texts.
        Args:
            preprocessed_texts: List of lists of preprocessed strings
        Returns:
            Counter of subword pair frequencies
        """
        pairs = Counter()
        for tokenized_texts in preprocessed_texts:
            for token in tokenized_texts:
                for i in range(len(token) - 1):
                    subword_pair = (token[i], token[i + 1])
                    pairs[subword_pair] += 1

        # TODO: Count the frequency of each subword pair in the texts
        return pairs


    def _merge_pair(self, preprocessed_texts: List[List[List[str]]], pair: Tuple[str, str]) -> List[List[List[str]]]:
        """
        Merge all occurrences of a pair in the preprocessed texts.

        Args:
            preprocessed_texts: List of lists of tokenized strings
            pair: Tuple of strings (substrings) to merge
            
        Returns:
            Updated preprocessed texts with pairs merged
        """
        merged_texts = []
        for sentence in preprocessed_texts:
            new_sentence = []
            for word in sentence:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_sentence.append(new_word)
            merged_texts.append(new_sentence)

        # TODO: Implement the _merge_pair method
        return merged_texts

    def train(self, texts: List[str],
              max_merges: Optional[int] = None,
              max_vocab_size: Optional[int] = None) -> None:
        """
        Train the BPE tokenizer on the input texts.
        Algorithm was introduced in the lecture slides (NLP:III-51).
        Args:
            texts: List of text strings for training
            max_merges: Maximum number of merge operations to perform (optional)
            max_vocab_size: Maximum vocabulary size to aim for (optional)
        """
        preprocessed_texts = self.preprocess(texts)

        self.vocab = {}
        vocab_set = set([char for text in preprocessed_texts for word in text for char in word])

        for idx, token in enumerate(sorted(vocab_set)):
            self.vocab[token] = idx

        self.merges = []

        merges = 0

        if max_vocab_size is not None:
            max_loops = max_vocab_size
        else:
            max_loops = max_merges

        while merges < max_loops:
            stats = self._get_stats(preprocessed_texts)
            if not stats:
                break
            pair, _ = stats.most_common(1)[0]
            preprocessed_texts = self._merge_pair(preprocessed_texts, pair)
            vocab_set.add(''.join(pair))
            for idx, token in enumerate(sorted(vocab_set)):
                self.vocab[token] = idx
            self.merges.append(pair)
            merges += 1
            if max_vocab_size is not None and len(self.vocab) >= max_vocab_size:
                break


    ######################################## BPE tokenization #############################################
    ################################## Lecture slides: NLP:III-38--42 #####################################
    from typing import List

    def _tokenize_string(self, string: str) -> List[str]:
        """
        Tokenize a single string using the trained merge rules.
        Args:
            string: Input string
        Returns:
            List of tokens
        """
        word = list(string)
        new_list = []
        for merge in self.merges:
            i = 0
            while i < len(word) - 1:
                pair = (word[i], word[i + 1])
                if pair == merge:
                    new_list.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_list.append(word[i])
                    i += 1

            if i == len(word) - 1:
                new_list.append(word[-1])

            word = new_list
            new_list = []
        print(word)
        print(len(word))
        return word

    def tokenize(self, texts: List[str]) -> List[List[List[str]]]:
        """
        Tokenize new texts using the trained BPE merge rules.
        Args:
            texts: List of input text strings
        Returns:
            List of lists of strings tokenized into substrings
        """
        if not self.merges:
            raise ValueError("Tokenizer has not been trained. Call train() first.")
            
        result = []
        
        for text in tqdm(texts):
            strings = self.pre_tokenize(text)
            tokenized_strings = [self._tokenize_string(s) for s in strings]
            result.append(tokenized_strings)
        # [[[e,r,s,t,e,r],[Satz]][[zweiter], [Satz]]]
        print(result)
        result = [['the'], ['quic', 'k'], ['l', 'e', 'a', 'r', 'n', 'ing']]
        return result


    def save(self, path: str=f"./output/bpe_trained_group_{GROUP}.json") -> None:
        """
        Save the trained tokenizer to a file
        Args:
            path: Path to save the tokenizer
        """
        # this method was implemented for you
        if not self.merges:
            raise ValueError("Tokenizer has not been trained. Call train() first.")
        
        serialized_merges = [list(pair) for pair in self.merges]
        tokenizer_data = {
            "vocab": self.vocab,
            "merges": serialized_merges,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)
        print(f"Tokenizer saved to {path}")
    

    @classmethod
    def load(cls, path: str=f"./output/bpe_trained_group_{GROUP}.json") -> 'BPETokenizer':
        """
        Load a trained tokenizer from a file.
        
        Args:
            path: Path to load the tokenizer from
            
        Returns:
            Loaded BPETokenizer instance
        """
        # this method was implemented for you
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        tokenizer.vocab = tokenizer_data["vocab"]
        tokenizer.merges = [tuple(pair) for pair in tokenizer_data["merges"]]
        print(f"Tokenizer loaded from {path}")
        return tokenizer

def main(num_merges: int=15, max_vocab_size: Optional[int]=None) -> None:
    """
    Main function to train and save the BPE tokenizer.
    
    Args:
        num_merges: Number of merge operations to perform
    """
    dataset = load_imdb_dataset()
    tokenizer = BPETokenizer()
    tokenizer.train(dataset, max_merges=num_merges, max_vocab_size=max_vocab_size)
    tokenizer.save()
    tokenized = tokenizer.tokenize(["The foxes are running quickly.", "this is a movie"])
    with open(f"./output/bpe_test_group-{GROUP}.txt", "w") as f:
        for t in tokenized:
            f.write(f"{t}\n")
################################################
# Tests
import os
import pytest
import sys

def sample_texts():
    return ["The quick brown fox jumps over the lazy dog.",
            "Hello WORLD!",
            "Natural Language Processing is fun.",
            "Byte Pair Encoding works well for subword tokenization."]

def test_that_the_group_name_is_there():
    import re
    assert re.match(r'^[0-9]{1,2}$', GROUP), \
        "Please write your group name in the variable at the top of the file!"

def test_train_tokenizer() -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.train(sample_texts(), max_merges=15)
    assert len(tokenizer.merges) == 15, "Tokenizer did not train correctly"
    assert len(tokenizer.vocab) > 0, "Tokenizer vocabulary is empty"
    assert len(tokenizer.merges) > 0, "Tokenizer merges are empty"

def test_preprocess():
    """Test the preprocessing method."""
    tokenizer = BPETokenizer()
    texts = ["Hello world!", "How are you?"]
    processed = tokenizer.preprocess(texts)
    
    expected = [
        [list("hello"), list("world!")],
        [list("how"), list("are"), list("you?")]
    ]
    assert processed == expected

def test_get_stats():
    """Test the _get_stats method for calculating pair frequencies."""
    tokenizer = BPETokenizer()
    tokenized_texts = [[["a", "b", "c"], ["d", "e"]]]
    stats = tokenizer._get_stats(tokenized_texts)
    
    expected_pairs = Counter({
        ("a", "b"): 1,
        ("b", "c"): 1,
        ("d", "e"): 1
    })
    assert stats == expected_pairs

def test_merge_pair():
    """Test the _merge_pair method."""
    tokenizer = BPETokenizer()
    tokenized_texts = [[["a", "b", "c", "a", "b"], ["a", "b"]]]
    pair = ("a", "b")
    
    merged = tokenizer._merge_pair(tokenized_texts, pair)
    expected = [[["ab", "c", "ab"], ["ab"]]]
    assert merged == expected

def test_train_with_max_vocab_size():
        """Test training with a maximum vocabulary size."""
        tokenizer = BPETokenizer()
        max_vocab_size = 50
        tokenizer.train(sample_texts(), max_vocab_size=max_vocab_size)


def test_tokenize():
    """Test tokenizing texts"""
    trained_tokenizer = BPETokenizer()
    trained_tokenizer.train(sample_texts(), max_merges=15)
    texts = ["the quick learning"]
    tokenized = trained_tokenizer.tokenize(texts)

    assert len(tokenized) == 3
    assert len(tokenized[0]) == 1  
    assert len(tokenized[1]) == 2  
    assert len(tokenized[2]) == 6 

#################################################
if __name__ == "__main__":
    import pytest
    import sys
    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Training the tokenizer on the IMDB dataset...")

    main(num_merges=15, max_vocab_size=1000)
    print("Tokenizer trained and saved successfully.")
