"""Unit tests for agent nodes — no network calls."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lg_custom_rag_agent.agent import (
    END,
    GradeDocuments,
    generate_answer,
    generate_query_or_respond,
    grade_documents,
    rewrite_question,
    route_to_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    messages,
    rewrite_count: int = 0,
) -> dict:
    return {"messages": list(messages), "rewrite_count": rewrite_count}


# ---------------------------------------------------------------------------
# route_to_tools
# ---------------------------------------------------------------------------

class TestRouteToTools:
    def test_routes_to_tools_when_tool_calls(self):
        state = _make_state([
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "retrieve_blog_posts", "args": {}}]),
        ])
        assert route_to_tools(state) == "tools"

    def test_routes_to_end_without_tool_calls(self):
        state = _make_state([
            HumanMessage(content="hi"),
            AIMessage(content="hello!"),
        ])
        assert route_to_tools(state) == END


# ---------------------------------------------------------------------------
# grade_documents
# ---------------------------------------------------------------------------

class TestGradeDocuments:
    def test_yes_returns_generate_answer(self):
        state = _make_state([
            HumanMessage(content="question"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "retrieve_blog_posts", "args": {}}]),
            ToolMessage(content="relevant doc", tool_call_id="1"),
        ])
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = GradeDocuments(binary_score="yes")
        mock_model.with_structured_output.return_value = mock_structured

        with patch("lg_custom_rag_agent.agent._get_grader_model", return_value=mock_model):
            assert grade_documents(state) == "generate_answer"

    def test_no_returns_rewrite_question(self):
        state = _make_state([
            HumanMessage(content="question"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "retrieve_blog_posts", "args": {}}]),
            ToolMessage(content="irrelevant doc", tool_call_id="1"),
        ])
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = GradeDocuments(binary_score="no")
        mock_model.with_structured_output.return_value = mock_structured

        with patch("lg_custom_rag_agent.agent._get_grader_model", return_value=mock_model):
            assert grade_documents(state) == "rewrite_question"

    def test_loop_cap_returns_generate_answer(self):
        state = _make_state(
            [
                HumanMessage(content="question"),
                AIMessage(content="", tool_calls=[{"id": "1", "name": "retrieve_blog_posts", "args": {}}]),
                ToolMessage(content="doc", tool_call_id="1"),
            ],
            rewrite_count=2,
        )
        assert grade_documents(state) == "generate_answer"

    def test_fallback_on_structured_output_failure(self):
        state = _make_state([
            HumanMessage(content="question"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "retrieve_blog_posts", "args": {}}]),
            ToolMessage(content="doc", tool_call_id="1"),
        ])
        mock_model = MagicMock()
        mock_model.with_structured_output.side_effect = RuntimeError("structured output failed")
        mock_raw = MagicMock()
        mock_raw.content = "yes"
        mock_model.invoke.return_value = mock_raw

        with patch("lg_custom_rag_agent.agent._get_grader_model", return_value=mock_model):
            assert grade_documents(state) == "generate_answer"


# ---------------------------------------------------------------------------
# rewrite_question
# ---------------------------------------------------------------------------

class TestRewriteQuestion:
    def test_returns_human_message_and_increments(self):
        state = _make_state(
            [HumanMessage(content="original question")],
            rewrite_count=0,
        )
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="improved question")

        with patch("lg_custom_rag_agent.agent._get_response_model", return_value=mock_model):
            result = rewrite_question(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert result["messages"][0].content == "improved question"
        assert result["rewrite_count"] == 1

    def test_increments_existing_count(self):
        state = _make_state(
            [HumanMessage(content="q")],
            rewrite_count=1,
        )
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="improved")

        with patch("lg_custom_rag_agent.agent._get_response_model", return_value=mock_model):
            result = rewrite_question(state)

        assert result["rewrite_count"] == 2


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    def test_returns_model_response(self):
        state = _make_state([
            HumanMessage(content="question"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "retrieve_blog_posts", "args": {}}]),
            ToolMessage(content="some context", tool_call_id="1"),
        ])
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="the answer")

        with patch("lg_custom_rag_agent.agent._get_response_model", return_value=mock_model):
            result = generate_answer(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "the answer"


# ---------------------------------------------------------------------------
# generate_query_or_respond
# ---------------------------------------------------------------------------

class TestGenerateQueryOrRespond:
    def test_returns_model_response(self):
        state = _make_state([HumanMessage(content="hello")])
        mock_model = MagicMock()
        mock_model.bind_tools.return_value.invoke.return_value = AIMessage(content="hi there")

        with patch("lg_custom_rag_agent.agent._get_response_model", return_value=mock_model):
            result = generate_query_or_respond(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "hi there"
