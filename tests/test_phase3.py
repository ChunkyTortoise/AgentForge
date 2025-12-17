import pytest
from core.agents import ResearchAgent
from unittest.mock import MagicMock, patch

def test_scrape_tool_registered():
    """Verify scrape_web_page tool is registered in ResearchAgent."""
    agent = ResearchAgent(provider="gemini")
    tool_names = [t.name for t in agent.tools]
    assert "scrape_web_page" in tool_names
    assert "duckduckgo_search" in tool_names

@patch("requests.get")
def test_scrape_logic(mock_get):
    """Verify scrape logic (mocked)."""
    # Create a dummy response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body><h1>Test Title</h1><p>Test Content</p></body></html>"
    mock_get.return_value = mock_response

    # Instantiate agent to get the tool
    agent = ResearchAgent(provider="gemini")
    scrape_tool = next(t for t in agent.tools if t.name == "scrape_web_page")
    
    result = scrape_tool.run("http://example.com")
    
    assert "Test Title" in result
    assert "Test Content" in result
    # Ensure cleanup happened (no html tags ideally, but simple text extraction)
    assert "<h1>" not in result

if __name__ == "__main__":
    test_scrape_tool_registered()
    test_scrape_logic()
    print("Phase 3 tests passed!")
