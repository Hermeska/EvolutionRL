"""Small future adapter shared by local sampling backends."""

import asyncio
from concurrent.futures import Future as ConcurrentFuture


class LocalFuture:
    """APIFuture-compatible wrapper around an immediate value or worker future."""

    def __init__(self, value):
        self._value = value

    def result(self):
        if isinstance(self._value, ConcurrentFuture):
            return self._value.result()
        return self._value

    async def result_async(self):
        if isinstance(self._value, ConcurrentFuture):
            return await asyncio.wrap_future(self._value)
        return self._value
