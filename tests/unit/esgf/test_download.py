"""Test 1esmvalcore.esgf._download`."""
from esmvalcore.esgf._download import Queue


def test_queue():
    """Test that Queue correctly tracks unfinished work."""
    queue = Queue()
    assert queue.unfinished_tasks == 0
    queue.put_nowait('a')
    assert queue.unfinished_tasks == 1
    queue.put_nowait('b')
    assert queue.unfinished_tasks == 2
    queue.task_done()
    assert queue.unfinished_tasks == 1
    queue.task_done()
    assert queue.unfinished_tasks == 0


def test_queue_put_back():
    """Test that Queue correctly tracks unfinished work."""
    queue = Queue()
    queue.put_nowait('a')
    assert queue.unfinished_tasks == 1
    item = queue.get_nowait()
    assert queue.unfinished_tasks == 1
    queue.put_back(item)
    assert queue.unfinished_tasks == 1
