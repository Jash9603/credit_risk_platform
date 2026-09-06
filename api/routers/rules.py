import json

from fastapi import APIRouter, HTTPException

from api.schemas import Rule, RuleCondition, RulesResponse
from src.rules.rule_miner import format_rule
from src.utils.config import settings

router = APIRouter(tags=["rules"])


@router.get("/rules", response_model=RulesResponse)
def get_rules():
    rules_path = settings.artifacts_path / "rules.json"
    if not rules_path.exists():
        raise HTTPException(status_code=404, detail="Rules not generated yet — run `python -m src.rules.rule_miner`.")

    raw_rules = json.loads(rules_path.read_text(encoding="utf-8"))
    rules = [
        Rule(
            conditions=[RuleCondition(**c) for c in r["conditions"]],
            support=r["support"],
            observed_default_rate=r["observed_default_rate"],
            lift_vs_average=r["lift_vs_average"],
            readable=format_rule(r),
        )
        for r in raw_rules
    ]
    return RulesResponse(rules=rules)
