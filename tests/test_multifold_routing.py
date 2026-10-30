import pandas as pd
from core.agent import _split_multifold_question, ask_agent


def test_splits_clear_semicolon_separated_tasks():
    parts = _split_multifold_question('What columns are available?; Check data quality')
    assert len(parts) == 2


def test_agent_combines_independent_basic_questions(monkeypatch):
    df = pd.DataFrame({'Sales':[10,20], 'Region':['East','West']})
    monkeypatch.setattr('core.agent.GEMINI_API_KEY', '')
    result = ask_agent('What columns are available?; Check data quality', df)
    assert result['tool'] == 'multi_analysis'
    assert result['result']['task_count'] == 2
    assert result['result']['successful_tasks'] == 2


def test_agent_splits_conjunction_followed_by_new_action(monkeypatch):
    df = pd.DataFrame({'Sales': [10, 20], 'Region': ['East', 'West']})
    result = ask_agent('What columns are available? and also check data quality', df)
    assert result['tool'] == 'multi_analysis'
    assert result['result']['task_count'] == 2
    assert result['result']['successful_tasks'] == 2


def test_multi_task_result_reports_partial_failures(monkeypatch):
    df = pd.DataFrame({'Sales': [10, 20], 'Region': ['East', 'West']})
    original = __import__('core.agent', fromlist=['ask_agent']).ask_agent
    # Verify combiner contract independently of LLM behaviour.
    from core.result_validation import combine_task_results
    combined = combine_task_results('two tasks', [
        {'question': 'known', 'status': 'success', 'result': {'total': 30}},
        {'question': 'unknown', 'status': 'error', 'message': 'missing data'},
    ])
    assert combined['status'] == 'success'
    assert combined['result']['partial_result']
    assert combined['result']['failed_tasks'] == 1
