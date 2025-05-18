#!/usr/bin/python3
from tqdm import tqdm
import re
import json
import os
from collections import  Counter
from typing import List,  Tuple, Optional


GROUP = "17"  # TODO: write in your group number

def load_imdb_dataset(file_path: str="./imdb_neu.txt", small_dataset=False) -> List[str]:
    """ This function loads the IMDB dataset from the txt file.
    Args:
        file_path (str): The path to the json file.
        p (int): Percentage of the dataset to use.
    Returns:
        list: A list of texts from the dataset.
    """
    # this function was implemented for you
    with open(file_path, "r", encoding="utf-8") as f:
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
        # text = re.sub(r'[^A-Za-z0-9\s]', '', text) # removes all special characters
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
            sentence = []
            # Use custom tokenizer to tokenize the text into strings
            pre_tokenized_text = self.pre_tokenize(text)
            for pre_tokenized in pre_tokenized_text:
                word = list(pre_tokenized) # [w, o, r, d],
                for letter in word:
                    sentence.append(letter)
            # TODO: implement the _split_into_characters method
            # For each word in the pre_tokenized text, split it into characters
            # and add it to the tokenized_texts list
            preprocessed_texts.append(sentence)

        return preprocessed_texts

    def _get_stats_with_threshold(self, preprocessed_texts: List[List[str]], threshold: int) -> set[str]:
        """
        Count the frequency of all adjacent subword pairs in the preprocessed texts, and return only
        those pairs whose frequency is greater than or equal to the given threshold.
        """
        stats = self._get_stats(preprocessed_texts)
        vocab_set = set()

        for pair, count in stats.items():
            if count >= threshold:
                vocab_set.add(pair)

        return vocab_set


    def _get_stats(self, preprocessed_texts: List[List[str]]) -> Counter:
        """
        Count subword pair frequencies in the preprocessed texts.
        Args:
            preprocessed_texts: List of lists of preprocessed strings
        Returns:
            Counter of subword pair frequencies
        """
        pairs = Counter()
        for tokenized_text in preprocessed_texts:
            for i in range(len(tokenized_text) - 1):
                subword_pair = (tokenized_text[i], tokenized_text[i + 1])
                pairs[subword_pair] += 1
        # TODO: Count the frequency of each subword pair in the texts
        return pairs

    def _merge_pair(self, 
                    preprocessed_texts: List[List[str]], 
                    pair: Tuple[str, str]) -> List[List[str]]:
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
            i = 0
            while i < len(sentence):
                if i < len(sentence) - 1 and (sentence[i], sentence[i + 1]) == pair:
                    new_sentence.append(sentence[i] + sentence[i + 1])
                    i += 2
                else:
                    new_sentence.append(sentence[i])
                    i += 1
            merged_texts.append(new_sentence)
        print(pair)
        return merged_texts

        # TODO: Implement the _merge_pair method

    def train(self, 
              texts: List[str], 
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
        # Lecture slides: NLP:III-43--51
        # 1. Create an initial tokenization of a training corpus 
        # + 3. Split each token into symbols; 
        preprocessed_texts = self.preprocess(texts)

        # 3. initialize vocabulary V with individual characters
        self.vocab = {}
        vocab_set = set([char for text in preprocessed_texts for word in text for char in word])

        # assign IDs to each token in the vocabulary
        for idx, token in enumerate(sorted(vocab_set)):
            self.vocab[token] = idx

        # initialize merge rules list
        self.merges = []
        merges = 0

        if max_vocab_size is not None:
            rounds = max_vocab_size - len(vocab_set)
        elif max_merges is not None:
            rounds = max_merges
        else:
            rounds = 5000

        print(rounds)
        # TODO: Implement the BPE training loop
        while merges < rounds:
            stats = self._get_stats(preprocessed_texts)
            if not stats:
                break

            pair, _ = stats.most_common(1)[0]

            # Neues Token aus dem Paar bilden und prüfen
            new_token = ''.join(pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            # Merge durchführen
            print(merges)
            preprocessed_texts = self._merge_pair(preprocessed_texts, pair)
            self.merges.append(pair)
            merges += 1

            # Frühzeitiger Abbruch, falls max_vocab_size erreicht ist
            if max_vocab_size is not None and len(self.vocab) >= max_vocab_size:
                break




    ######################################## BPE tokenization #############################################
    ################################## Lecture slides: NLP:III-38--42 #####################################
    def _tokenize_string(self, string: str) -> List[str]:
        """
        Tokenize a single string using the trained merge rules.
        Args:
            string: Input string
        Returns:
            List of tokens
        """

        # Lecture slides: NLP:III-38--42
        # TODO: Implement the _tokenize_string method

        sentence = []
        pre_tokenized_text = self.pre_tokenize(string)
        for pre_tokenized in pre_tokenized_text:
            word = list(pre_tokenized)  # [w, o, r, d],
            for letter in word:
                sentence.append(letter)

        new_list = []
        for merge in self.merges:
            i = 0
            while i < len(sentence) - 1:
                pair = (sentence[i], sentence[i + 1])
                if pair == merge:
                    new_list.append(sentence[i] + sentence[i + 1])
                    i += 2
                else:
                    new_list.append(sentence[i])
                    i += 1

            if i == len(sentence) - 1:
                new_list.append(sentence[-1])

            sentence = new_list
            new_list = []
        return sentence


    def tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize new texts using the trained BPE merge rules.
        Args:
            texts: List of input text strings
        Returns:
            List of lists of strings tokenized into substrings
        """
        # this method was implemented for you
        if not self.merges:
            raise ValueError("Tokenizer has not been trained. Call train() first.")
            
        result = []
        
        for text in tqdm(texts):
            # strings = self.pre_tokenize(text)
            # The call above is commented out because `pre_tokenize` returns a list of words,
            # but `_tokenize_string` expects a string, not a list.

            # tokenized_strings = [self._tokenize_string(s) for s in strings]
            # This line is commented out because it applies `_tokenize_string` to each individual word,
            # resulting in a list of tokenized words instead of a full tokenized sentence.

            tokenized_strings = self._tokenize_string(text) # we call the function on the whole sentence
            result.append(tokenized_strings)


        return result


    def save(self, path: str=f"./output/bpe_group_{GROUP}.json") -> None:
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)
        print(f"Tokenizer saved to {path}")
    

    @classmethod
    def load(cls, path: str=f"./output/bpe_group_{GROUP}.json") -> 'BPETokenizer':
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

def main(num_merges: int=1000, 
         max_vocab_size: Optional[int]=None, 
         small_dataset: bool=False) -> None:
    """
    Main function to train and save the BPE tokenizer.
    
    Args:
        num_merges: Number of merge operations to perform
    """
    dataset = load_imdb_dataset(small_dataset=small_dataset)
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
    return ["the quick brown fox jumps over the lazy dog.",
            "bpe works well for subword tokenization."]

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
    processed = tokenizer.preprocess(sample_texts())
    
    expected = [['t', 'h', 'e', 'q', 'u', 'i', 'c', 'k', 'b', 'r', 'o', 'w', 'n', 'f', 'o', 'x', 
                 'j', 'u', 'm', 'p', 's', 'o', 'v', 'e', 'r', 't', 'h', 'e', 'l', 'a', 'z', 'y', 'd', 'o', 'g', '.'], 
                 ['b', 'p', 'e', 'w', 'o', 'r', 'k', 's', 'w', 'e', 'l', 'l', 'f', 'o', 'r', 
                  's', 'u', 'b', 'w', 'o', 'r', 'd', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n', '.']]
    assert processed == expected

def test_get_stats():
    """Test the _get_stats method for calculating pair frequencies."""
    tokenizer = BPETokenizer()
    # tokenized_texts = [["a", "b", "c"], ["d", "e"]], [["b", "c"], ["d", "e"]]
    tokenized_texts = tokenizer.preprocess(sample_texts())
    # tokenized_texts = [[['h', 'e', 'l', 'l', 'o'], ['w', 'o', 'r', 'l', 'd', '!']], [['h', 'o', 'w'], ['a', 'r', 'e'], ['y', 'o', 'u', '?']]]
    
    stats = tokenizer._get_stats(tokenized_texts)
    
    expected_pairs = Counter({('o', 'r'): 3, ('t', 'h'): 2, ('h', 'e'): 2, ('f', 'o'): 2, ('e', 'l'): 2, ('w', 'o'): 2, 
                              ('e', 'q'): 1, ('q', 'u'): 1, ('u', 'i'): 1, ('i', 'c'): 1, ('c', 'k'): 1, ('k', 'b'): 1, 
                              ('b', 'r'): 1, ('r', 'o'): 1, ('o', 'w'): 1, ('w', 'n'): 1, ('n', 'f'): 1, ('o', 'x'): 1, 
                              ('x', 'j'): 1, ('j', 'u'): 1, ('u', 'm'): 1, ('m', 'p'): 1, ('p', 's'): 1, ('s', 'o'): 1, 
                              ('o', 'v'): 1, ('v', 'e'): 1, ('e', 'r'): 1, ('r', 't'): 1, ('l', 'a'): 1, ('a', 'z'): 1, 
                              ('z', 'y'): 1, ('y', 'd'): 1, ('d', 'o'): 1, ('o', 'g'): 1, ('g', '.'): 1, ('b', 'p'): 1, 
                              ('p', 'e'): 1, ('e', 'w'): 1, ('r', 'k'): 1, ('k', 's'): 1, ('s', 'w'): 1, ('w', 'e'): 1, 
                              ('l', 'l'): 1, ('l', 'f'): 1, ('r', 's'): 1, ('s', 'u'): 1, ('u', 'b'): 1, ('b', 'w'): 1, 
                              ('r', 'd'): 1, ('d', 't'): 1, ('t', 'o'): 1, ('o', 'k'): 1, ('k', 'e'): 1, ('e', 'n'): 1, 
                              ('n', 'i'): 1, ('i', 'z'): 1, ('z', 'a'): 1, ('a', 't'): 1, ('t', 'i'): 1, ('i', 'o'): 1, 
                              ('o', 'n'): 1, ('n', '.'): 1})
    assert stats == expected_pairs

def test_merge_pair():
    """Test the _merge_pair method."""
    tokenizer = BPETokenizer()
    tokenized_texts = [['t', 'h', 'e', 'q', 'u', 'i', 'c', 'k', 'b', 'r', 'o', 'w', 'n', 'f', 'o', 'x', 
                 'j', 'u', 'm', 'p', 's', 'o', 'v', 'e', 'r', 't', 'h', 'e', 'l', 'a', 'z', 'y', 'd', 'o', 'g', '.'], 
                 ['b', 'p', 'e', 'w', 'o', 'r', 'k', 's', 'w', 'e', 'l', 'l', 'f', 'o', 'r', 
                  's', 'u', 'b', 'w', 'o', 'r', 'd', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n', '.']]
    pair = ("o", "r")
    
    merged = tokenizer._merge_pair(tokenized_texts, pair)
    #print('MERGED:', merged)
    expected = [['t', 'h', 'e', 'q', 'u', 'i', 'c', 'k', 'b', 'r', 'o', 'w', 'n', 'f', 'o', 'x', 
                 'j', 'u', 'm', 'p', 's', 'o', 'v', 'e', 'r', 't', 'h', 'e', 'l', 'a', 'z', 'y', 'd', 'o', 'g', '.'], 
                 ['b', 'p', 'e', 'w', 'or', 'k', 's', 'w', 'e', 'l', 'l', 'f', 'or', 
                  's', 'u', 'b', 'w', 'or', 'd', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n', '.']]
    assert merged == expected

def test_train_with_max_vocab_size():
        """Test training with a maximum vocabulary size."""
        tokenizer = BPETokenizer()
        max_vocab_size = 30
        tokenizer.train(sample_texts(), max_vocab_size=max_vocab_size)
        assert len(tokenizer.vocab) == max_vocab_size, "Tokenizer did not limit vocabulary size correctly"


def test_tokenize():
    """Test tokenizing texts"""
    tokenizer = BPETokenizer()
    tokenizer.merges = [('t', 'h'), ('h', 'e'), ('n', 'g'), ('i', 'ng'), ('c', 'k'), ('th', 'e')]

    # trained_tokenizer.train(sample_texts(), max_merges=5)
    texts = ["the quick learning", 'hello word']
    tokenized = tokenizer.tokenize(texts)
    expected = [['the', 'q', 'u', 'i', 'ck', 'l', 'e', 'a', 'r', 'n', 'ing'], 
                ['he', 'l', 'l', 'o', 'w', 'o', 'r', 'd']]
    assert tokenized == expected, f"Tokenization failed: {tokenized} != {expected}"

#################################################
if __name__ == "__main__":
    import pytest
    import sys
    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Training the tokenizer on the IMDB dataset...")

    main(num_merges=10, max_vocab_size=None) # TODO: change the number of merges and/or vocab size
    print("Tokenizer trained and saved successfully.")
