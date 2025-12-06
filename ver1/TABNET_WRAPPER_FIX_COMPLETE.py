"""
TabNetWrapper 완전 수정 버전
=========================
이 코드를 복사해서 TABNET_ENHANCED_MODEL.py의 393-448줄을 교체하세요.
"""

from sklearn.base import BaseEstimator, RegressorMixin
from pytorch_tabnet.tab_model import TabNetRegressor
import torch


class TabNetWrapper(BaseEstimator, RegressorMixin):
    """TabNet을 sklearn 스타일로 래핑"""
    
    def __init__(self, tabnet_model=None):
        self.tabnet_model = tabnet_model
        self.model = tabnet_model
    
    def _more_tags(self):
        """sklearn이 regressor로 인식하도록"""
        return {'regressor': True}
    
    @property
    def _estimator_type(self):
        """sklearn의 check_estimator가 regressor로 인식"""
        return "regressor"
    
    def fit(self, X, y):
        # TabNet 모델이 없으면 새로 생성 (clone 시)
        if self.model is None:
            self.model = TabNetRegressor(
                n_d=32,
                n_a=32,
                n_steps=5,
                gamma=1.5,
                lambda_sparse=1e-4,
                momentum=0.3,
                mask_type='entmax',
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size": 10, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                verbose=0,
                seed=42
            )
        
        # y가 2D가 아니면 2D로 변환
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        self.model.fit(
            X, y,
            max_epochs=100,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
            eval_metric=['rmse']
        )
        return self
    
    def predict(self, X):
        pred = self.model.predict(X)
        # 1D로 변환 (sklearn stacking이 요구)
        if len(pred.shape) > 1:
            pred = pred.ravel()
        return pred
    
    def get_params(self, deep=True):
        """sklearn 호환을 위한 get_params"""
        return {"tabnet_model": self.tabnet_model}
    
    def set_params(self, **params):
        """sklearn 호환을 위한 set_params"""
        if "tabnet_model" in params:
            self.tabnet_model = params["tabnet_model"]
            self.model = params["tabnet_model"]
        return self
    
    def score(self, X, y):
        """sklearn 호환을 위한 score (R² 계산)"""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))
