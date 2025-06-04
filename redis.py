## docker run -p 6379:6379 --name redis-mod redislabs/redismod

import redis

pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
r = redis.Redis(connection_pool=pool)
r.set('food', 'mutton', ex=3)
print(r.get('food'))


r.mset({"key1": "Hello", "key2": "World"})
values = r.mget(["key1", "key2", "nonexisting"])
print(values)
