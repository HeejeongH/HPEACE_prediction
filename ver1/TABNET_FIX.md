# TabNetWrapper ValueError 수정 가이드

## 오류 내용
```
ValueError: The estimator TabNetWrapper should be a regressor.
```

## 원인
`TabNetWrapper` 클래스가 sklearn의 regressor로 제대로 인식되지 않음

## 해결 방법

`TABNET_ENHANCED_MODEL.py` 파일의 `TabNetWrapper` 클래스를 다음과 같이 수정:

### 수정 전 (393-448줄)
```python
class TabNetWrapper(BaseEstimator, RegressorMixin):
    """TabNet을 sklearn 스타일로 래핑"""
    def __init__(self, tabnet_model=None):
        self.tabnet_model = tabnet_model
        self.model = tabnet_model
```

### 수정 후
```python
class TabNetWrapper(BaseEstimator, RegressorMixin):
    """TabNet을 sklearn 스타일로 래핑"""
    
    # sklearn이 regressor로 인식하도록 명시
    _estimator_type = "regressor"
    
    def __init__(self, tabnet_model=None):
        self.tabnet_model = tabnet_model
        self.model = tabnet_model
```

## 전체 수정된 클래스 코드

```python
class TabNetWrapper(BaseEstimator, RegressorMixin):
    """TabNet을 sklearn 스타일로 래핑"""
    
    # sklearn이 regressor로 인식하도록 명시
    _estimator_type = "regressor"
    
    def __init__(self, tabnet_model=None):
        self.tabnet_model = tabnet_model
        self.model = tabnet_model
    
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
```

## 수정 방법 (텍스트 에디터 사용)

1. `ver1/src/TABNET_ENHANCED_MODEL.py` 파일 열기
2. 393줄의 `class TabNetWrapper(BaseEstimator, RegressorMixin):` 찾기
3. 395줄 `def __init__` 위에 다음 2줄 추가:
   ```python
   # sklearn이 regressor로 인식하도록 명시
   _estimator_type = "regressor"
   ```
4. 저장 후 다시 실행

## 확인

수정 후 다시 실행:
```bash
python run_training.py safe
```
