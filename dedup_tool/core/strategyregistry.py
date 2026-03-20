from typing import Type

from dedup_tool.core.strategy import DedupStrategy


class StrategyRegistry:
    _strategies = {}

    @classmethod
    def register(cls, name: str):
        def decorator(strategy_cls: Type[DedupStrategy]):
            cls._strategies[name] = strategy_cls
            return strategy_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[DedupStrategy]:
        return cls._strategies[name]
