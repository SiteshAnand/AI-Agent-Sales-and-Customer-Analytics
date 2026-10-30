"""Association-rule mining for transaction baskets using a bounded Apriori search."""
from __future__ import annotations
import itertools
import re
import pandas as pd

MAX_CANDIDATES = 100_000


def _find_column(df: pd.DataFrame, aliases: list[str]):
    normalized = {re.sub(r'[^a-z0-9]+', '', str(c).casefold()): c for c in df.columns}
    for alias in aliases:
        key = re.sub(r'[^a-z0-9]+', '', alias.casefold())
        if key in normalized:
            return normalized[key]
    return None


def _label(items: tuple[str, ...]) -> str:
    return " + ".join(items)


def market_basket_analysis(
    df: pd.DataFrame,
    transaction_column: str | None = None,
    product_column: str | None = None,
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    min_lift: float = 1.0,
    max_rules: int = 100,
    max_itemset_size: int = 3,
) -> dict:
    """Calculate support, confidence, and lift for rules with up to N items.

    Apriori pruning avoids testing combinations whose subsets are not frequent.
    The search is bounded by ``MAX_CANDIDATES`` to protect the app from an
    accidental combinatorial explosion on very high-cardinality datasets.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError('Upload a non-empty transaction dataset first.')
    if not 0 < min_support <= 1 or not 0 < min_confidence <= 1:
        raise ValueError('min_support and min_confidence must be greater than 0 and at most 1.')
    if min_lift < 0 or type(max_rules) is not int or not 1 <= max_rules <= 1000:
        raise ValueError('min_lift must be non-negative and max_rules between 1 and 1000.')
    if type(max_itemset_size) is not int or not 2 <= max_itemset_size <= 4:
        raise ValueError('max_itemset_size must be an integer between 2 and 4.')

    transaction_column = transaction_column or _find_column(df, ['Transaction_ID','Transaction ID','Order_ID','Order ID','Invoice','InvoiceNo','Basket_ID','Receipt_ID'])
    product_column = product_column or _find_column(df, ['Product_Name','Product Name','Product','Item','Item_Name','SKU','Product_ID'])
    if transaction_column not in df.columns or product_column not in df.columns:
        raise ValueError('Market Basket Analysis requires transaction/order ID and product/item columns. Provide one row per item in a transaction.')

    data = df[[transaction_column, product_column]].dropna().copy()
    data[transaction_column] = data[transaction_column].astype(str).str.strip()
    data[product_column] = data[product_column].astype(str).str.strip()
    data = data[(data[transaction_column] != '') & (data[product_column] != '')].drop_duplicates()
    if data.empty:
        raise ValueError('No valid transaction-product pairs were found.')
    baskets = data.groupby(transaction_column)[product_column].agg(lambda s: frozenset(s))
    n = len(baskets)
    if n < 2:
        raise ValueError('At least two valid transactions are required for association analysis.')

    item_counts: dict[tuple[str, ...], int] = {}
    single_counts: dict[str, int] = {}
    for basket in baskets:
        for item in basket:
            single_counts[item] = single_counts.get(item, 0) + 1
            item_counts[(item,)] = single_counts[item]

    min_count = max(1, int(-(-min_support * n // 1)))  # ceil(min_support * n)
    frequent: dict[tuple[str, ...], int] = {
        itemset: count for itemset, count in item_counts.items() if count / n >= min_support
    }
    all_frequent = dict(frequent)
    previous_level = {tuple(sorted(itemset)) for itemset in frequent}
    candidates_examined = len(previous_level)

    for size in range(2, max_itemset_size + 1):
        if not previous_level:
            break
        candidates = set()
        ordered_previous = sorted(previous_level)
        for left_index, left in enumerate(ordered_previous):
            for right in ordered_previous[left_index + 1:]:
                candidate = tuple(sorted(set(left) | set(right)))
                if len(candidate) != size:
                    continue
                if all(tuple(subset) in previous_level for subset in itertools.combinations(candidate, size - 1)):
                    candidates.add(candidate)
                if candidates_examined + len(candidates) > MAX_CANDIDATES:
                    raise ValueError(
                        'Market Basket search exceeded its safety limit. Increase minimum support or reduce max_itemset_size.'
                    )
        candidates_examined += len(candidates)
        current_level = {}
        for candidate in candidates:
            count = sum(set(candidate).issubset(basket) for basket in baskets)
            if count / n >= min_support:
                current_level[candidate] = int(count)
        all_frequent.update(current_level)
        previous_level = set(current_level)

    rules = []
    for itemset, joint_count in all_frequent.items():
        if len(itemset) < 2:
            continue
        joint_support = joint_count / n
        for antecedent_size in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, antecedent_size):
                consequent = tuple(item for item in itemset if item not in antecedent)
                antecedent_count = all_frequent.get(tuple(sorted(antecedent)), 0)
                consequent_count = all_frequent.get(tuple(sorted(consequent)), 0)
                if antecedent_count == 0 or consequent_count == 0:
                    continue
                confidence = joint_count / antecedent_count
                lift = confidence / (consequent_count / n)
                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        'Antecedent': _label(tuple(antecedent)),
                        'Consequent': _label(tuple(consequent)),
                        'Transactions_Together': int(joint_count),
                        'Support': round(joint_support, 6),
                        'Confidence': round(confidence, 6),
                        'Lift': round(lift, 6),
                        'Itemset_Size': len(itemset),
                    })

    rules.sort(key=lambda r: (r['Lift'], r['Confidence'], r['Support']), reverse=True)
    return {
        'transaction_column_used': transaction_column,
        'product_column_used': product_column,
        'transaction_count': int(n),
        'unique_products': int(len(single_counts)),
        'rule_count_before_limit': len(rules),
        'rules': rules[:max_rules],
        'parameters': {'min_support': min_support, 'min_confidence': min_confidence,
                       'min_lift': min_lift, 'max_itemset_size': max_itemset_size,
                       'candidate_safety_limit': MAX_CANDIDATES},
        'warnings': [
            f'Association rules searched itemsets up to size {max_itemset_size}; candidate search is safety-bounded.',
            'Association is not causation and does not prove that one product causes another purchase.',
        ],
    }
