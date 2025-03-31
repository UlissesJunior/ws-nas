from transformers.models.deepseek.modeling_deepseek import (
    DeepseekForSequenceClassification,
    DeepseekForMultipleChoice,
)

from model_wrapper.mask import mask_deepseek  # Você precisará implementar esta função
from search_spaces import (
    SmallSearchSpace,
    LayerSearchSpace,
    FullSearchSpace,
    MediumSearchSpace,
)


class DeepseekSuperNetMixin:
    search_space = None
    handles = None

    def select_sub_network(self, sub_network_config):
        head_mask, ffn_mask = self.search_space.config_to_mask(sub_network_config)
        head_mask = head_mask.to(device="cuda", dtype=self.dtype)
        ffn_mask = ffn_mask.to(device="cuda", dtype=self.dtype)
        self.handles = mask_deepseek(self.deepseek, ffn_mask, head_mask)

    def reset_super_network(self):
        for handle in self.handles:
            handle.remove()


class DeepseekSuperNetMixinLAYERSpace(DeepseekSuperNetMixin):
    @property
    def search_space(self):
        return LayerSearchSpace(self.config)


class DeepseekSuperNetMixinMEDIUMSpace(DeepseekSuperNetMixin):
    @property
    def search_space(self):
        return MediumSearchSpace(self.config)


class DeepseekSuperNetMixinLARGESpace(DeepseekSuperNetMixin):
    @property
    def search_space(self):
        return FullSearchSpace(self.config)


class DeepseekSuperNetMixinSMALLSpace(DeepseekSuperNetMixin):
    @property
    def search_space(self):
        return SmallSearchSpace(self.config)


class SuperNetDeepseekForSequenceClassificationSMALL(
    DeepseekForSequenceClassification, DeepseekSuperNetMixinSMALLSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetDeepseekForMultipleChoiceSMALL(
    DeepseekForMultipleChoice, DeepseekSuperNetMixinSMALLSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetDeepseekForSequenceClassificationLAYER(
    DeepseekForSequenceClassification, DeepseekSuperNetMixinLAYERSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetDeepseekForMultipleChoiceLAYER(
    DeepseekForMultipleChoice, DeepseekSuperNetMixinLAYERSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetDeepseekForSequenceClassificationMEDIUM(
    DeepseekForSequenceClassification, DeepseekSuperNetMixinMEDIUMSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetDeepseekForMultipleChoiceMEDIUM(
    DeepseekForMultipleChoice, DeepseekSuperNetMixinMEDIUMSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetDeepseekForSequenceClassificationLARGE(
    DeepseekForSequenceClassification, DeepseekSuperNetMixinLARGESpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetDeepseekForMultipleChoiceLARGE(
    DeepseekForMultipleChoice, DeepseekSuperNetMixinLARGESpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)