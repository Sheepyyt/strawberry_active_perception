# AGX NERO v1.11 本地安全补丁

本目录保存对固定上游版本 `22a9cf6c5ad2fd2e0743531936bc5dab007fa5bc`
的本地修改，避免更新或重建子模块时丢失已经过真机验证的安全行为。

补丁包含：NERO v1.11 无反馈启动修复、`move_home` 控制门、真实 SDK 电子阻尼
急停及其锁存、准确的使能状态，以及对应测试和文档。它不包含任何上游版本升级。

```bash
cd /home/yyt/strawberry_active_perception
./vendor_patches/agx_arm_ros/apply_checked.sh
```

脚本只接受上述精确 commit。当前工作树已经包含补丁时只做验证；工作树存在其它
修改时会拒绝覆盖。补丁使用零上下文格式，但只有在 commit 和补丁 SHA 都完全一致时才会
应用，因此不会把它静默套到未知版本。补丁 SHA256：

```text
394542caff32f5d147a7a33059a0fa078e96416276f0a61d9fed62b48e8104d4
```
