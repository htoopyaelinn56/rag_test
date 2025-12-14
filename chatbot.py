from db_service import retrieve_context

for i in retrieve_context("features"):
    print(i)
    print('-----')