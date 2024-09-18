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

hours_of_clockface = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
]

hour_intervals = [
    "0 hours",
    "1 hour",
    "2 hours",
    "3 hours",
    "4 hours",
    "5 hours",
    "6 hours",
    "7 hours",
    "8 hours",
    "9 hours",
    "10 hours",
    "11 hours",
]

class HoursOfClockfaceTask:
    def __init__(self, device, model_name="mistral", n_devices=None):
        self.device = device

        self.model_name = model_name

        self.n_devices = n_devices

        # Tokens we expect as possible answers. Best of these can optionally be saved (as opposed to best logit overall)
        self.allowable_tokens = hours_of_clockface

        self.prefix = f"{BASE_DIR}{model_name}_hours_of_clockface/"
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)

        self.num_tokens_in_answer = 1

        # this is only for the figure
        self.prediction_names = ["hour of clockface"]

        # Leave tokens until difference commented out since we never want to plot them
        if model_name == "mistral":
            self.token_map = {
                # 0: "<s>",
                # 1: "Let",
                # 2: "apostrophe",
                # 3: "s after apostrophe",
                # 4: "do",
                # 5: "some",
                # 6: "days (first)",
                # 7: "of",
                # 8: "the",
                # 9: "week",
                # 10: "math",
                # 11: "<Period>",
                12: "<Num duration days>",
                13: "days (second)",
                14: "from",
                15: "<Start day of week>",
                16: "is",
                17: "<Target day of week>",
            }
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
            #     14: "<Start day of week>",
            #     15: "is",
            #     16: "<Target day of week>",
            # }
            self.token_map = { 0: '<|begin_of_text|>', 1: 'Let', 2: "'s", 3: 'Ġdo', 4: 'Ġsome', 5: 'Ġclock', 6: 'Ġmath', 7: '.', 8: 'ĠThe', 9: 'Ġhour', 10: 'Ġhand', 11: 'Ġis', 12: 'Ġpointing', 13: 'Ġto', 14: 'Ġthe', 15: 'Ġ', 16: '12', 17: '.', 18: 'ĠIn', 19: 'Ġ', 20: '7', 21: 'Ġhours', 22: ',', 23: 'Ġit', 24: 'Ġwill', 25: 'Ġbe', 26: 'Ġpointing', 27: 'Ġto', 28: 'Ġthe' }

        self.b_token = 20
        self.a_token = 16
        self.before_c_token = 28

        # (Friendly name, index into Problem.info)
        self.how_to_color = [
            ("target_hour", 2),
            ("start_hour", 0),
            ("duration_hours", 1),
        ]

        # Used for figures folder
        self.name = f"{model_name}_hours_of_clockface"

        self._lazy_model = None

    def _get_prompt(self, starting_hour_int, num_hours_int):
        starting_hour_str = hours_of_clockface[starting_hour_int]
        num_hours_str = hour_intervals[num_hours_int]
        # prompt = f"Let's do some hours of the week math. {num_hours_str} from {starting_hour_str} is"
        prompt = f"Let's do some clock math. The hour hand is pointing to the {starting_hour_str}. In {num_hours_str} hours, it will be pointing to the"

        correct_answer_int = (starting_hour_int + num_hours_int) % 12
        correct_answer_str = hours_of_clockface[correct_answer_int]

        # TODO: Should we distinguish between carrys and not in the correct answer?
        return prompt, correct_answer_str, correct_answer_int

    def generate_problems(self):
        np.random.seed(42)
        problems = []
        for starting_hour in range(12):
            for num_hours in range(1, 12):
                prompt, correct_answer_str, correct_answer_int = self._get_prompt(
                    starting_hour_int=starting_hour, num_hours_int=num_hours
                )
                problems.append(
                    Problem(
                        prompt,
                        correct_answer_str,
                        (starting_hour, num_hours, correct_answer_int),
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
        important_tokens = [16, 20, 28]
        if self.model_name == "llama":
            for i in range(len(important_tokens)):
                important_tokens[i] -= 1
        return important_tokens


# %%

if __name__ == "__main__":
    task = HoursOfClockfaceTask(device, model_name="llama")
    # task = HoursOfClockfaceTask(device, model_name="mistral")


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
