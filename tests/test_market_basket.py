import pandas as pd
import pytest
from core.market_basket import market_basket_analysis


def sample():
    return pd.DataFrame({
        'Order_ID': ['1','1','2','2','3','3','4','4'],
        'Product': ['A','B','A','B','A','C','B','C'],
    })


def test_market_basket_known_support_confidence_lift():
    result = market_basket_analysis(sample(), min_support=0.25, min_confidence=0.1, min_lift=0)
    rule = next(r for r in result['rules'] if r['Antecedent'] == 'A' and r['Consequent'] == 'B')
    assert result['transaction_count'] == 4
    assert rule['Support'] == 0.5
    assert rule['Confidence'] == round(2/3, 6)
    assert rule['Lift'] == round((2/3) / (3/4), 6)


def test_market_basket_deduplicates_same_item_within_transaction():
    df = pd.DataFrame({'Order_ID':['1','1','1','2','2'], 'Product':['A','A','B','A','B']})
    result = market_basket_analysis(df, min_support=0.5, min_confidence=0.1, min_lift=0)
    assert result['transaction_count'] == 2
    assert result['rules'][0]['Transactions_Together'] == 2


def test_market_basket_requires_columns():
    with pytest.raises(ValueError, match='transaction/order ID'):
        market_basket_analysis(pd.DataFrame({'x':[1,2]}))


def test_market_basket_validates_thresholds():
    with pytest.raises(ValueError, match='min_support'):
        market_basket_analysis(sample(), min_support=0)


def test_market_basket_supports_three_item_rules():
    df = pd.DataFrame({
        'Order_ID': ['1','1','1','2','2','2','3','3','3','4','4'],
        'Product': ['A','B','C','A','B','C','A','B','D','A','C'],
    })
    result = market_basket_analysis(df, min_support=0.5, min_confidence=0.5, min_lift=0, max_itemset_size=3)
    assert any(rule['Itemset_Size'] == 3 for rule in result['rules'])


def test_market_basket_rejects_excessive_itemset_size():
    with pytest.raises(ValueError, match='max_itemset_size'):
        market_basket_analysis(sample(), max_itemset_size=5)
