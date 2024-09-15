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

hours_of_day = [
    "00:00",
    "01:00",
    "02:00",
    "03:00",
    "04:00",
    "05:00",
    "06:00",
    "07:00",
    "08:00",
    "09:00",
    "10:00",
    "11:00",
    "12:00",
    "13:00",
    "14:00",
    "15:00",
    "16:00",
    "17:00",
    "18:00",
    "19:00",
    "20:00",
    "21:00",
    "22:00",
    "23:00",
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
    "12 hours",
    "13 hours",
    "14 hours",
    "15 hours",
    "16 hours",
    "17 hours",
    "18 hours",
    "19 hours",
    "20 hours",
    "21 hours",
    "22 hours",
    "23 hours",
    "24 hours"
]


class DaysOfWeekTask:
    def __init__(self, device, model_name="mistral", n_devices=None):
        self.device = device

        self.model_name = model_name

        self.n_devices = n_devices

        # Tokens we expect as possible answers. Best of these can optionally be saved (as opposed to best logit overall)
        self.allowable_tokens = hours_of_day

        self.prefix = f"{BASE_DIR}{model_name}_hours_of_day/"
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)

        self.num_tokens_in_answer = 1

        self.prediction_names = ["day of week"]

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
            self.token_map = {
                0: '<|begin_of_text|>',
                1: 'Let',
                2: "'s",
                3: 'Ġdo',
                4: 'Ġsome',
                5: 'Ġclock',
                6: 'Ġmath',
                7: '.',
                8: 'ĠRight',
                9: 'Ġnow',
                10: 'Ġit',
                11: 'Ġis',
                12: 'Ġ',
                13: '<Start hour of day>',
                14: ':',
                15: '00',
                16: '.',
                17: 'ĠIn',
                18: 'Ġ',
                19: '<Num duration hours>',
                20: 'Ġhours',
                21: 'Ġit',
                22: 'Ġwill',
                23: 'Ġbe',
                24: '<Target hour of day>',
            }

        self.b_token = 19
        self.a_token = 13
        self.before_c_token = 23

        # (Friendly name, index into Problem.info)
        self.how_to_color = [
            ("target_hour", 2),
            ("start_hour", 0),
            ("duration_hours", 1),
        ]

        # Used for figures folder
        self.name = f"{model_name}_hours_of_day"

        self._lazy_model = None

    def _get_prompt(self, starting_hour_int, num_hours_int):
        starting_hour_str = hours_of_day[starting_hour_int]
        num_hours_str = hour_intervals[num_hours_int]
        prompt = f"Let's do some clock math. Right now it is {starting_hour_str}. In {num_hours_str} hours it will be"

        correct_answer_int = (starting_hour_int + num_hours_int) % 24
        correct_answer_str = hours_of_day[correct_answer_int]

        # TODO: Should we distinguish between carrys and not in the correct answer?
        return prompt, correct_answer_str, correct_answer_int

    def generate_problems(self):
        np.random.seed(42)
        problems = []
        for starting_day in range(24):
            for num_days in range(1, 25):
                prompt, correct_answer_str, correct_answer_int = self._get_prompt(
                    starting_hour_int=starting_day, num_hours_int=num_days
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
        important_tokens = [13, 19, 23, 24]
        if self.model_name != "llama":
            raise Exception("Only llama supported!")
        return important_tokens

# %%

if __name__ == "__main__":
    task = DaysOfWeekTask(device, model_name="llama")
    # task = DaysOfWeekTask(device, model_name="mistral")


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
