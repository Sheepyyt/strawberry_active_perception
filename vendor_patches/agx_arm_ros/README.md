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
修改时会拒绝覆盖。补丁 SHA256：

```text
80af58642bf9fd58610056343875cd45b55ce62724fa747d104c1d30972b6cd2
```
