import os

import pytest

from kep.compatibility_graph import Digraph


@pytest.fixture
def graph_instance():
    """Fixture that loads the test graph before each test function."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    arc_file = os.path.join(test_dir, "tests_data", "arc_test_digraph.txt")

    g = Digraph()
    if os.path.exists(arc_file):
        g.load(arc_file)
    else:
        pytest.fail(f"Graph file not found at: {arc_file}")
    return g

def test_load_graph(graph_instance):
    """Test if the graph loads the correct number of nodes and edges."""
    g = graph_instance

    assert g.nv == 4, "The number of vertices should be 4"
    assert g.ne == 5, "The number of edges should be 5"

def test_adjacency_list(graph_instance):
    """Test if the connections (who points to whom) are correct."""
    g = graph_instance

    # Does 0 point to 1?
    assert 1 in g.adjList[0], "Node 0 should point to node 1"
    # Does 1 point to 2 and 3?
    assert set(g.adjList[1]) == {2, 3}, "Node 1 should point to nodes 2 and 3"
    # Does 2 point to 0?
    assert 0 in g.adjList[2], "Node 2 should point to node 0"
    # Does 3 point to 0?
    assert 0 in g.adjList[3], "Node 3 should point to node 0"

def test_incidence_list(graph_instance):
    """Test the incidence list (who points to me)."""
    g = graph_instance

    # Who points to 0 are 2 and 3
    assert set(g.incList[0]) == {2, 3}, "Node 0 should receive pointers from nodes 2 and 3"
    # Who points to 1 is 0
    assert 0 in g.incList[1], "Node 1 should receive a pointer from node 0"

def test_load_file_not_found():
    """Test if loading a non-existent file raises FileNotFoundError."""
    g = Digraph()
    with pytest.raises(FileNotFoundError):
        g.load("non_existent_file.txt")

def test_load_empty_file(tmp_path):
    """Test if loading an empty file raises ValueError."""
    d = tmp_path / "empty.txt"
    d.write_text("")  # Create empty file

    g = Digraph()
    with pytest.raises(ValueError, match="File is empty or invalid"):
        g.load(str(d))

def test_load_malformed_file(tmp_path):
    """Test ValueError when file has fewer lines than declared in header"""
    d = tmp_path / "malformed.txt"
    # Header says 3 node and 2 edges, but we provide 0 edge lines
    d.write_text("3,2\n")

    g = Digraph()
    with pytest.raises(ValueError, match="Warning: End of file reached"):
        g.load(str(d))

def test_load_malformed_lines(tmp_path):
    """Test Skipping malformed lines"""
    d = tmp_path / "malformed.txt"
    # Header: 3 nodes, 2 edges.
    # Line 2 is malformed (only one value), so it should be skipped.
    content = "3,2\n0,1,1\n0\n"
    d.write_text(content)

    g = Digraph()
    g.load(str(d))

    # We expect 0->1 to be loaded
    assert 1 in g.adjList[0], "Node 0 point to node 1 should be loaded"

def test_show_lists(graph_instance, capsys):
    """Test Print methods"""
    g = graph_instance

    g.showIncList()
    captured = capsys.readouterr()
    assert "Printing Incoming List" in captured.out

    g.showAdjList()
    captured = capsys.readouterr()
    assert "Printing Adjacency List" in captured.out

def test_reverse(graph_instance):
    """Test Reverse method"""
    g = graph_instance
    assert g.inverted is False

    g.reverse()

    assert g.inverted is True
    # Check if verts are sorted reverse (3, 2, 1, 0)
    assert g.verts == [3, 2, 1, 0]

    # Check if an adjacency list is sorted reverse
    # Node 1 pointed to {2, 3}. In reverse sort, it should appear as [3, 2]
    assert g.adjList[1] == [3, 2]

def test_to_dot_file(graph_instance, tmp_path):
    """Test export to DOT file"""
    g = graph_instance
    output_file = tmp_path / "graph.dot"

    g.to_dot_file(str(output_file))

    assert output_file.exists()

    content = output_file.read_text()
    assert "digraph G {" in content
    assert '"0" -> "1";' in content
    assert "}" in content
