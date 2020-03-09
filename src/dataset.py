import numpy as np
import torch
import transformers
from nltk.tokenize import RegexpTokenizer


class IMDBDataset:

    def __init__(self, review, label):
        self.__review = review.values
        self.__label = label.values
        self.__tokenizer = transformers.BertTokenizer.from_pretrained(
            "../input/",
            do_lower_case=True
        )
        self.__max_length = 512

    def get_max_len(self):
        return self.__max_length

    def get_tokenizer(self):
        return self.__tokenizer

    def get_review(self):
        return self.__review

    def get_label(self):
        return self.__label

    def __len__(self):
        return len(self.get_review())

    def __getitem__(self, item):
        text = str(self.get_review()[item])

        tokenizer = RegexpTokenizer(r'\w+')
        text_cleaned = tokenizer.tokenize(text)

        encoded_sequence = self.get_tokenizer().encode_plus(
            text=text_cleaned,
            max_length=self.get_max_len(),
            add_special_tokens=True
        )

        inputs_ids_array = np.zeros(self.get_max_len())
        attention_mask_array = np.zeros(self.get_max_len())
        token_type_ids_array = np.zeros(self.get_max_len())
        offset = len(encoded_sequence['input_ids'])

        inputs_ids_array[:offset] = encoded_sequence['input_ids']
        attention_mask_array[:offset] = encoded_sequence['attention_mask']
        token_type_ids_array[:offset] = encoded_sequence['token_type_ids']

        return [
            torch.tensor(data=inputs_ids_array, dtype=torch.int64),
            torch.tensor(data=attention_mask_array, dtype=torch.int64),
            torch.tensor(data=token_type_ids_array, dtype=torch.int64),
            torch.tensor(data=self.get_label()[item], dtype=torch.float64)
        ]

