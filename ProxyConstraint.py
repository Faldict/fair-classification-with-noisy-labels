import numpy as np 
import pandas as pd

from fairlearn.reductions import ConditionalSelectionRate

_GROUP_ID = "group_id"
_EVENT = "event"
_LABEL = "label"
_LOSS = "loss"
_PREDICTION = "pred"
_ALL = "all"
_SIGN = "sign"
_DIFF = "diff"

class ProxyEqualizedOdds(ConditionalSelectionRate):
    def __init__(self, error_rate=[[0.3, 0.3], [0.0, 0.0]]):
        super().__init__()
        self.error_rate = error_rate
    
    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y,
                          event=pd.Series(y).apply(lambda y: _LABEL + "=" + str(y)),
                          **kwargs)

    def gamma(self, predictor):
        """Calculate the degree to which constraints are currently violated by the predictor."""
        pred = predictor(self.X)
        self.tags[_PREDICTION] = pred
        expect_event = self.tags.groupby(_EVENT).mean()
        expect_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).mean()

        neg = expect_group_event.loc[(expect_group_event.index.get_level_values(_GROUP_ID) == 0.0)].groupby([_EVENT]).mean()
        pos = expect_group_event.loc[(expect_group_event.index.get_level_values(_GROUP_ID) == 1.0)].groupby([_EVENT]).mean()
        # print(pos)
        expect_group_event.loc[('label=1.0', 1), 'pred'] = (1 - self.error_rate[1][0]) * pos.loc['label=1.0', 'pred'] + self.error_rate[1][0] * pos.loc['label=0.0', 'pred']
        expect_group_event.loc[('label=0.0', 1), 'pred'] = (1 - self.error_rate[1][1]) * pos.loc['label=0.0', 'pred'] + self.error_rate[1][1] * pos.loc['label=1.0', 'pred']

        expect_group_event.loc[('label=1.0', 0), 'pred'] = (1 - self.error_rate[0][0]) * neg.loc['label=1.0', 'pred'] + self.error_rate[0][0] * neg.loc['label=0.0', 'pred']
        expect_group_event.loc[('label=0.0', 0), 'pred'] = (1 - self.error_rate[0][1]) * neg.loc['label=0.0', 'pred'] + self.error_rate[0][1] * neg.loc['label=1.0', 'pred']

        expect_event = expect_group_event.groupby(_EVENT).mean()
        expect_group_event[_DIFF] = expect_group_event[_PREDICTION] - expect_event[_PREDICTION]

        # expect_group_event[_DIFF] = expect_group_event[_PREDICTION] - expect_event[_PREDICTION]
        g_unsigned = expect_group_event[_DIFF]
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+", "-"],
                             names=[_SIGN, _EVENT, _GROUP_ID])
        self._gamma_descr = str(expect_group_event[[_PREDICTION, _DIFF]])
        return g_signed