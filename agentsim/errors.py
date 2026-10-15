"""统一错误类型：所有面向用户的消息必须可读（人和 AI 都要看得懂）。"""


class WorldError(Exception):
    """世界文件解析/校验/编译错误。message 应包含：位置 + 期望 + 实际值。"""
