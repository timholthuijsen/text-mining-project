# -*- coding: utf-8 -*-

import clean_data as cd

# Get a list of available databases
dbs = cd.getDatabases()

# Clean the first one
clean_db = cd.cleanFile(dbs[0])

# Print 10 lines
cd.head(clean_db, 10)