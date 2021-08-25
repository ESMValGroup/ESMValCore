"""
Parse and display test memory.

Uses pytest-monitor plugin from https://github.com/CFMTech/pytest-monitor
Lots of other metrics can be read from the file via sqlite parsing.,
currently just MEM_USAGE (RES memory, in MB).
"""
import sqlite3
import sys
from operator import itemgetter


def _parse_pymon_database():
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect(".pymon")
    cur = con.cursor()

    # The result of a "cursor.execute" can be iterated over by row
    # first look at memory
    print("Looking for tests that exceed 1GB resident memory.")
    big_mem_tests = []
    for row in cur.execute('select ITEM, MEM_USAGE from TEST_METRICS;'):
        test_name, memory_used = row[0], row[1]
        if memory_used > 1000.:  # test result in RES mem in MB
            print("Test name / memory (MB)")
            print(test_name, memory_used)
        elif memory_used > 4000.:
            big_mem_tests.append((test_name, memory_used))

    # then look at total time (in seconds)
    # (user time is availbale too via USER_TIME, kernel time via KERNEL_TIME)
    timed_tests = []
    sq_command = 'select ITEM, ITEM_VARIANT, TOTAL_TIME from TEST_METRICS;'
    for row in cur.execute(sq_command):
        test_name, test_var, time_used = row[0], row[1], row[2]
        timed_tests.append((test_name, test_var, time_used))

    timed_tests = sorted(timed_tests, reverse=True, key=itemgetter(2))
    hundred_slowest_tests = timed_tests[0:100]
    print("List of 100 slowest tests (test name, test variant: duration [s])")
    if hundred_slowest_tests:
        for test_name, test_var, test_duration in hundred_slowest_tests:
            print(test_name + " " + test_var + ": " + "%.2f" % test_duration)
    else:
        print("Could not retrieve test timing data.")

    # Be sure to close the connection
    con.close()

    # Throw a sys exit so test fails if we have >4GB  tests
    if big_mem_tests:
        print("Some tests exceed 4GB of RES memory, look into them!")
        print(big_mem_tests)
        sys.exit(1)


if __name__ == '__main__':
    _parse_pymon_database()
