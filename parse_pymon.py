"""
Parse and display test memory.

Uses pytest-monitor plugin from https://github.com/CFMTech/pytest-monitor
Lots of other metrics can be read from the file via sqlite parsing.,
currently just MEM_USAGE (RES memory, in MB).
"""
import sqlite3

# Create a SQL connection to our SQLite database
con = sqlite3.connect(".pymon")

cur = con.cursor()

# The result of a "cursor.execute" can be iterated over by row
print("Looking for tests that exceed 1GB resident memory.")
for row in cur.execute('select ITEM, MEM_USAGE from TEST_METRICS;'):
    test_name, memory_used = row[0], row[1]
    if memory_used > 1000.:  # test result in RES mem in MB
        print("Test name / memory (MB)")
        print(test_name, memory_used)
    else:
        print("All our tests need less than 1GB resident memory, phew.")

# Be sure to close the connection
con.close()
