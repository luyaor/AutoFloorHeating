## 可能的问题现象及其原因

- debug 绘制出来白屏
    - 很可能是某个绘制的点坐标极大，一般是交点出现问题。

## BUG

- 蓝色起点飘了
- build_linked_g2 给出的 sets_di 不一定是逆时针，可能会连出交的

## TODO

- 外部给的数据必须 Assertion
- M1 名字搞错了算了

```
got pt dir
for pt0, dir0 in out:
    seg0 = pt0, pt0'nxt
    inner_line0 = seg0'inner_line(sug)
    获得与 inner_line0 相交时刻
    对于线段
        - 不断远离的线段：从距离大于等于 sug 开始允许
        - 其他线段：一旦距离小于 sug 就禁止
```

- g3 图已经建立了一些点和边。(G2Node)
- solve() 传入一个环
- spiral() 传入一个 Polygon 返回一个 spiral 图，可能添加了一些新点
- 需要知道 spiral 图（外围）与 g3 传入环的对应点关系,
- spiral
    - 先按需求创建 g3 新点
        - spiral 图需要有 id. id 对应 g3 (list index) or new
- return a dict: 