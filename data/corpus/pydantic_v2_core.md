# Pydantic v2 Core Architecture & Validation Guide

## 1. Overview and Core Engine
Pydantic v2 is a rewrite of Pydantic with the core validation logic implemented in Rust via `pydantic-core`. This yields a 5x to 50x performance improvement over Pydantic v1.

## 2. Defining Models and Fields
Models inherit from `pydantic.BaseModel`. Fields are annotated with Python types and optionally configured using `Field`:
```python
from pydantic import BaseModel, Field, EmailStr

class UserModel(BaseModel):
    user_id: int = Field(..., gt=0, description="Unique primary key")
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    is_active: bool = True
    tags: list[str] = Field(default_factory=list)
```

## 3. Validators in Pydantic v2
Pydantic v2 deprecates `@validator` and `@root_validator` in favor of `@field_validator` and `@model_validator`.

### 3.1 Field Validators (`@field_validator`)
Field validators validate individual attributes:
```python
from pydantic import BaseModel, field_validator

class DocumentQuery(BaseModel):
    query_text: str
    top_k: int = 4

    @field_validator("query_text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query text cannot be empty or whitespace only")
        return cleaned

    @field_validator("top_k")
    @classmethod
    def validate_top_k_range(cls, v: int) -> int:
        if not (1 <= v <= 20):
            raise ValueError("top_k must be between 1 and 20")
        return v
```

### 3.2 Model Validators (`@model_validator`)
Model validators execute across multiple fields with either `mode='before'` (raw input dict) or `mode='after'` (typed model instance):
```python
from pydantic import BaseModel, model_validator

class SignupPayload(BaseModel):
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def verify_passwords_match(self) -> "SignupPayload":
        if self.password != self.confirm_password:
            raise ValueError("password and confirm_password must match")
        return self
```

## 4. Serialization and Dumps
In v2, `.dict()` is replaced by `.model_dump()`, and `.json()` is replaced by `.model_dump_json()`.
```python
user = UserModel(user_id=1, username="alex", email="alex@example.com")
data_dict = user.model_dump(exclude_unset=True)
json_str = user.model_dump_json(indent=2)
```
