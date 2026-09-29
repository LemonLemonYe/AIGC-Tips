# Hindsight 记忆系统技术解读

整理日期：2026-09-29。代码依据：[vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)，提交 `1e42702`。本文是静态代码阅读与设计分析，未部署服务、未进行成本或准确率实测。以下代码为关键摘录或明确标注的简化示意。

## 1. 核心判断

Hindsight 给 Agent 提供外部长期记忆：提取事实、建立检索索引、后台归纳，并在回答时查找证据。其“学习”主要是更新外部记忆，不是训练模型参数。

“事实提取＋向量检索＋摘要”本身并不独特。它的价值在于整合分层记忆、作用范围、增量处理、证据追踪和多路检索等工程机制。但它没有彻底解决记忆的适用性：**相似、相关，不代表当前有效，也不代表应该遵守。**

## 2. 写入到底消耗多少 token？

典型流程：

```text
原始文本 → 分块 → LLM 提取事实 → 向量与实体关系处理 → 数据库存储
                                                    ↓
                         后台检索旧归纳 → LLM 合并 → 按配置刷新专题页
```

数据库写入本身不收取模型 token 费用。消耗主要来自提取、归纳、专题页刷新，以及查询时的分析。相同信息可能多次进入模型上下文。

```text
总费用 = 所有 LLM 调用的输入、输出与缓存费用
       + 向量生成费用 + 重排序费用 + 数据库及计算资源费用
```

以下仅为假设示例，不是项目实测：

| 环节 | 输入 token | 输出 token |
|---|---:|---:|
| 原文分块提取，包含重复提示词 | 13,000 | 3,000 |
| 新事实与旧归纳合并 | 6,000 | 1,000 |
| 合计，未包括专题页刷新 | 19,000 | 4,000 |

若每百万输入、输出 token 单价分别为 P入、P出，则上述模型费用为 `0.019 × P入 + 0.004 × P出`。实际还受模型、缓存计费、事实密度、重试和历史归纳长度影响，不能仅凭原文字数给出固定价格。

### 代码：可跳过事实提取模型

文件：`hindsight-api-slim/hindsight_api/engine/retain/fact_extraction.py`，函数 `extract_facts_from_contents`。

```python
if config.retain_extraction_mode == "chunks":
    return _extract_facts_chunks(contents, config)
```

`chunks` 模式直接处理文本块，跳过这一阶段的 LLM 提取。后续向量、归纳或查询仍可能花钱，因此不等于全流程免费。另有批处理 API 路径；能否享受折扣取决于模型供应商及配置。

配置文件 `hindsight-api-slim/hindsight_api/config.py` 中：

```python
DEFAULT_RETAIN_CHUNK_SIZE = 3000  # 字符数，不是 token 数
DEFAULT_RETAIN_EXTRACTION_MODE = "concise"
DEFAULT_RETAIN_MAX_COMPLETION_TOKENS = 64000
```

`64000` 是输出上限配置，不是固定消耗。更详细的事实提取通常会增加输出及后续处理负担；更简洁的提取则可能丢失限定条件。

## 3. 如何把文本变成记忆？

同一提取文件中的事实构建逻辑，简化如下：

```python
extracted_fact = ExtractedFactType(
    fact_text=fact_from_llm.fact,
    fact_type=fact_from_llm.fact_type,
    entities=list(fact_from_llm.entities or []),
    occurred_start=...,        # 事件发生时间
    occurred_end=...,
    mentioned_at=content.event_date,  # 提及时间
    chunk_index=chunk_global_idx,     # 来源文本块
    metadata=content.metadata,
)
```

除了内容，还保存实体、时间、来源和附加信息。例如“9月29日说10月2日去泰安”，提及时间与事件时间不能混为一谈。

但结构化字段不能保证语义正确。例如“这张图不要红衣服”可能被错误提取成“用户不喜欢红衣服”，把局部指令升级成长期偏好。实体、时间、因果关系也可能提取错误。来源追踪让错误可排查，却不会自动消除错误。

## 4. 分层：事实、归纳与专题页

| 层次 | 保存什么 | 示例 |
|---|---|---|
| 原始材料 | 对话、文档及来源 | 用户多轮修改记录 |
| 事实／经历 | 可单独检索的信息单元 | 此图胸口不能露乳沟 |
| Observations | 多条事实形成的归纳及证据关联 | 用户在该项目中重视服饰一致性 |
| Mental Models | 围绕固定问题持续维护的专题答案 | 项目当前人物设定与视觉规范 |

这不是所有记忆必须逐层升级的流水线。上层是派生知识，应能回查下层证据。专题页预先生成，读取时可以直接读取数据库；生成与刷新仍有成本。

### 代码：先按作用范围分组，再归纳

文件：`hindsight-api-slim/hindsight_api/engine/consolidation/consolidator.py`。

```python
for m in memories:
    tag_key = _consolidation_batch_key(m)
    tag_groups.setdefault(tag_key, []).append(dict(m))

for group in tag_groups.values():
    grouped_batches.append(
        [group[i:i + llm_batch_size]
         for i in range(0, len(group), llm_batch_size)]
    )
```

先确定目标归纳范围，再在组内分批调用模型。这样能避免先把不同作用范围的材料混给模型，再试图通过输出过滤补救。范围由配置和输入决定，分组程序不能替用户判断标签是否正确。

随后 `_consolidate_batch_with_llm` 把新事实、相关旧归纳及来源事实放入模型输入，让模型提出创建、更新或删除归纳的决定。

### 代码：更新必须有真实来源

```python
source_mems = [
    mem_by_id[fid]
    for fid in update.source_fact_ids
    if fid in mem_by_id
]
if not source_mems:
    continue
```

后续还检查目标归纳是否在相应来源事实的召回范围内，以及来源是否仍存在。这限制了模型操作不存在或无关对象，但不证明“来源内容足以支持更新后的结论”。

`proof_count` 表示支持记录的数量，不是正确概率；同一错误被重复记录也可能增加计数。

## 5. 为什么向量匹配不够？它如何补强？

检索结合语义、关键词、图关系和时间等路径，再融合和重排序。

文件：`hindsight-api-slim/hindsight_api/engine/search/fusion.py`。

```python
rrf_scores[doc_id] += 1.0 / (k + rank)
```

默认 `k=60`。这是 Reciprocal Rank Fusion：候选在多路榜单中越靠前、被越多路径找到，融合分越高。使用名次，避免直接比较不同检索算法量纲不同的分数。

但“以前穿红衣”与“现在不要穿红衣”可能同时高度相关。融合只决定哪些信息值得阅读，不决定哪条有效。

`search/reranking.py` 提供 Cross-encoder 重排序，把问题和候选内容一起评分，并包含近期性、时间及证据数量等加权机制。它比独立向量比较更细致，但相关性仍不等于适用性或真实性。

`fusion.py` 还提供 `interleave_fusion`：轮流保留各路榜单的前列结果。代码说明了一个具体故障模式：后台归纳时，语义排名第一的近重复归纳可能被普通融合挤出预算，导致模型创建重复归纳。轮流融合用于缓解这种问题；不能据此说所有检索默认都使用它。

## 6. 防止错误应用：硬过滤必须在前

文件：`hindsight-api-slim/hindsight_api/engine/search/tags.py`。

| 匹配模式 | 语义 |
|---|---|
| any | 任一指定标签匹配，也包含无标签记忆 |
| all | 包含全部指定标签，也包含无标签记忆 |
| any_strict | 任一指定标签匹配，排除无标签记忆 |
| all_strict | 包含全部指定标签，排除无标签记忆 |
| exact | 标签集合完全相同 |

如果同时指定项目、角色和版本，`any` 可能只因为角色相同就选中其他版本。普通 strict 模式未提供标签时也不等于拒绝所有内容；`exact` 空范围有其特殊语义。因此这些参数不是完整权限系统，认证和访问范围仍需正确设计。

Bank 用来划分用户、Agent 或项目的记忆库。应用应先确定 bank 和作用范围，再做相关性检索，而不是让相似度决定项目、版本或身份。

## 7. Reflect：有证据不等于结论正确

Reflect 多轮调用检索工具后生成回答，本身是只读操作，不自动把回答写成新记忆。

文件：`hindsight-api-slim/hindsight_api/engine/reflect/agent.py`。

```python
has_gathered_evidence = (
    bool(available_memory_ids)
    or bool(available_mental_model_ids)
    or bool(available_observation_ids)
)
if not has_gathered_evidence and iteration < max_iterations - 1:
    # 要求先搜索信息
```

有剩余迭代机会时，阻止模型在未搜集证据的情况下直接结束。但这不是“证据不足则必定拒答”的硬保证；找到证据也不代表证据充分。

最终的引用过滤：

```python
used_memory_ids = [
    mid for mid in (args.get("memory_ids") or [])
    if mid in available_memory_ids
]
```

它确认引用 ID 来自本次检索集合，不能证明答案中的每项主张都被引用内容支持。准确说，这是引用来源集合校验，不是逐条逻辑验证。

## 8. 面向 AIGC 项目的补强建议

以下为设计建议，不表示 Hindsight 已自动实现这些完整规则。

重要设定应显式记录：

```json
{
  "project": "柠檬剑舞",
  "entity": "女主",
  "claim": "头发为银白色",
  "scope": "角色长期设定",
  "status": "confirmed",
  "version": 3,
  "source_type": "用户明确确认",
  "supersedes": "旧设定ID",
  "source_id": "原始确认记录ID"
}
```

处理分三步：

1. **资格过滤**：身份、项目、角色、版本、状态和权限是否允许使用。
2. **相关性检索**：在合格内容中用关键词、向量、关系等找候选。
3. **应用审查**：检查限定条件、矛盾与证据；冲突未解决时不悄悄选一个。

正式设定、单图临时要求、未确认创意、模型推断和已替代历史应分别管理。今天模型生成的草稿不能因时间更新就覆盖昨天用户确认的设定。来源、状态与范围通常比单纯的新旧排序更重要。

正式版本可继续放在 GitHub Markdown 中，记忆库用于检索相关决定和经验。

## 9. 如何实际验证成本与质量

先用代表性材料做小规模试验，分别记录提取、后台归纳、专题页刷新和查询的 token、缓存、重试及费用。必须等后台任务结束后再汇总，不能只看 retain 请求的前台用量。

代码已有用量记录与增量处理路径，但增量处理有适用条件，不能假设所有更新都会自动避免重复提取。

建议的成本策略：

- 完整保留原始资料，只提取值得长期使用的信息。
- 根据材料选择 chunks 或事实提取，不统一使用最昂贵路径。
- 正式设定变更或积累一定更新后再刷新专题页。
- 简单查询读取事实或已有专题页，复杂问题才用 Reflect。

建议用以下案例测错误应用率，而不只测“找没找到”：

| 测试输入 | 应有行为 |
|---|---|
| 同一角色的旧版和新版服装 | 当前任务使用已确认的适用版本 |
| “这张图不要红色” | 不升级为用户永久讨厌红色 |
| 两个项目都有同名角色 | 不跨项目混用设定 |
| 模型草稿与用户确认冲突 | 不让草稿自动覆盖确认内容 |
| 多次重复同一错误 | 不把记录数量当真实性 |
| 有相似材料但无充分证据 | 表达不确定或请求澄清 |

## 10. 结论

Hindsight 是较完整的记忆加工与检索基础设施。它的工程价值体现在分层、作用范围、来源追踪、多路检索、增量更新和后台处理，而不只是向量数据库。

**向量检索负责找候选；记忆是否有资格约束当前任务，需要明确的数据规则和应用逻辑。**
