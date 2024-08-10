# %%

import os
from utils import setup_notebook, BASE_DIR

setup_notebook()

import numpy as np
import transformer_lens
from task import Problem, get_acts, plot_pca, get_all_acts, get_acts_pca
from task import activation_patching


device = "cuda:4"
#
# %%

cardinal_dirs = [
    "North",
    "Northeast",
    "East",
    "Southeast",
    "South",
    "Southwest",
    "West",
    "Northwest",
]

degrees_rotation = [ str(i * 45) + ' degrees' for i in range(9) ]

class CardinalDirsClockwiseTask:
    def __init__(self, device, model_name="mistral", n_devices=None):
        print('CardinalDirsClockwiseTask.__init__')

        self.device = device

        self.model_name = model_name

        self.n_devices = n_devices

        # Tokens we expect as possible answers. Best of these can optionally be saved (as opposed to best logit overall)
        self.allowable_tokens = cardinal_dirs

        self.prefix = f"{BASE_DIR}{model_name}_cardinal_dirs_clockwise/"
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)

        self.num_tokens_in_answer = 1

        self.prediction_names = ["heading"]

        # Leave tokens until difference commented out since we never want to plot them
        if model_name == "mistral":
            # self.token_map = {
            #     # 0: "<s>",
            #     # 1: "Let",
            #     # 2: "apostrophe",
            #     # 3: "s after apostrophe",
            #     # 4: "do",
            #     # 5: "some",
            #     # 6: "days (first)",
            #     # 7: "of",
            #     # 8: "the",
            #     # 9: "week",
            #     # 10: "math",
            #     # 11: "<Period>",
            #     12: "<Num duration days>",
            #     13: "days (second)",
            #     14: "from",
            #     15: "<Start heading>",
            #     16: "is",
            #     17: "<Target heading>",
            # }
            self.token_map = {
                # 0: 'A'
                # 1: 'boat'
                # 2: 'is'
                # 3: 'cruising'
                # 4: 'with'
                # 5: 'a'
                # 6: 'heading'
                # 7: 'of'
                8: '<Start heading>',
                9: '.',
                10: 'It',
                11: 'turns',
                12: '<Num degrees rotation>',
                13: 'degrees',
                14: 'to',
                15: 'the',
                16: 'right',
                17: '.',
                18: 'It',
                19: 'is',
                20: 'now',
                21: 'cruising',
                22: 'with',
                23: 'a',
                24: 'heading',
                25: 'of',
                26: '<Target heading>',
            }
            self.b_token = 12
            self.a_token = 8
            self.before_c_token = 25
            # self.token_map = ['A', 'boat', 'is', 'cruising', 'with', 'a', 'heading', 'of', '<Start heading>', '.', 'It', 'turns', '<Num degrees rotation>', 'degrees', 'to', 'the', 'right', '.', 'It', 'is', 'now', 'cruising', 'with', 'a', 'heading', 'of', '<Target heading>']
        else:
            # self.token_map = {
            #     # 0: "<|begin_of_text|>",
            #     # 1: "Let",
            #     # 2: "apostrophe s",
            #     # 3: "do",
            #     # 4: "some",
            #     # 5: "days",
            #     # 6: "of",
            #     # 7: "the",
            #     # 8: "week",
            #     # 9: "math",
            #     # 10: "<Period>",
            #     11: "<Num duration days>",
            #     12: "days (second)",
            #     13: "from",
            #     14: "<Start heading>",
            #     15: "is",
            #     16: "<Target heading>",
            # }
            # self.token_map = ['<|begin_of_text|>', 'A', 'boat', 'is', 'cruising', 'with', 'a', 'heading', 'of', '<Start heading>', '.', 'It', 'turns', '', '<Num degrees rotation>', 'degrees', 'to', 'the', 'right', '.', 'It', 'is', 'now', 'cruising', 'with', 'a', 'heading', 'of', '<Target heading>']
            self.token_map = {
                # 0: '<|begin_of_text|>',
                # 1: 'A',
                # 2: 'boat',
                # 3: 'is',
                # 4: 'cruising',
                # 5: 'with',
                # 6: 'a',
                # 7: 'heading',
                # 8: 'of',
                9: '<Start heading>',
                10: '.',
                11: 'It',
                12: 'turns',
                13: '<Num degrees rotation>',
                14: 'degrees',
                15: 'to',
                16: 'the',
                17: 'right',
                18: '.',
                19: 'It',
                20: 'is',
                21: 'now',
                22: 'cruising',
                23: 'with',
                24: 'a',
                25: 'heading',
                26: 'of',
                27: '<Target heading>'
            }
            # self.b_token = 11 + (1 if model_name == "mistral" else 0)
            # self.a_token = 14 + (1 if model_name == "mistral" else 0)
            # self.before_c_token = 15 + (1 if model_name == "mistral" else 0)
            self.b_token = 13
            self.a_token = 9
            self.before_c_token = 26
        
        print('self.a_token', self.a_token)

        # (Friendly name, index into Problem.info)
        self.how_to_color = [
            ("target_day", 2),
            ("start_day", 0),
            ("duration_days", 1),
        ]

        # Used for figures folder
        self.name = f"{model_name}_cardinal_dirs"

        self._lazy_model = None

    def _get_prompt(self, starting_heading_int, degrees_rotation_int):
        starting_heading_str = cardinal_dirs[starting_heading_int]
        degrees_to_rotate = degrees_rotation[degrees_rotation_int]
        # prompt = f"Let's do some days of the week math. {num_days_str} from {starting_day_str} is"
        prompt = f"A boat is cruising with a heading of {starting_heading_str}. It turns {degrees_to_rotate} degrees to the right. It is now cruising with a heading of"

        correct_answer_int = (starting_heading_int + degrees_rotation_int) % 8
        correct_answer_str = cardinal_dirs[correct_answer_int]

        # TODO: Should we distinguish between carrys and not in the correct answer?
        return prompt, correct_answer_str, correct_answer_int

    def generate_problems(self):
        np.random.seed(42)
        problems = []
        for starting_day in range(8):
            for num_days in range(1, 9):
                prompt, correct_answer_str, correct_answer_int = self._get_prompt(
                    starting_heading_int=starting_day, degrees_rotation_int=num_days
                )
                problems.append(
                    Problem(
                        prompt,
                        correct_answer_str,
                        (starting_day, num_days, correct_answer_int),
                    )
                )

        np.random.shuffle(problems)
        return problems

    def get_model(self):
        if self.n_devices is None:
            self.n_devices = 1 if "llama" == self.model_name else 1
        if self._lazy_model is None:
            if self.model_name == "mistral":
                self._lazy_model = transformer_lens.HookedTransformer.from_pretrained(
                    "mistral-7b", device=self.device, n_devices=self.n_devices
                )
            elif self.model_name == "llama":
                self._lazy_model = transformer_lens.HookedTransformer.from_pretrained(
                    # "NousResearch/Meta-Llama-3-8B",
                    "meta-llama/Meta-Llama-3-8B",
                    device=self.device,
                    n_devices=self.n_devices,
                )
        return self._lazy_model

    def important_tokens(self):
        important_tokens = [12, 13, 14, 15, 16]
        if self.model_name == "llama":
            for i in range(len(important_tokens)):
                important_tokens[i] -= 1
        return important_tokens


# %%

if __name__ == "__main__":
    task = CardinalDirsClockwiseTask(device, model_name="llama")
    # task = CardinalDirsClockwiseTask(device, model_name="mistral")


# %%

if __name__ == "__main__":
    # Force generation of PCA k = 20
    for layer in range(33):
        for token in task.important_tokens():
            _ = get_acts_pca(task, layer=layer, token=token, pca_k=20)


# %%

if __name__ == "__main__":
    do_pca = True
    if do_pca:
        for token_location in task.important_tokens():
            # for normalize in [True, False]:
            #     for k in [2,3]:
            for normalize in [False]:
                for k in [2]:
                    plot_pca(
                        task,
                        token_location=token_location,
                        k=k,
                        normalize_rms=normalize,
                        include_embedding_layer=True,
                    )
# %%

if __name__ == "__main__":
    for layer_type in ["mlp", "attention", "resid"]:
        # for layer_type in ["attention"]:
        for keep_same_index in [0, 1]:
            activation_patching(
                task,
                keep_same_index=keep_same_index,
                num_chars_in_answer_to_include=0,
                num_activation_patching_experiments_to_run=20,
                layer_type=layer_type,
            )


# %%
