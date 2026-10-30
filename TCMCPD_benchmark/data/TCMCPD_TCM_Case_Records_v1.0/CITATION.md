# 引用信息 / Citation — TCMCPD

## 数据集本体（Dataset）

> Chen, J., Ma, Y., Zhao, B., Wang, R., Chen, G., et al. (2026).
> **TCMCPD: An Integrated TCM Case records and Pharmacology Database** [Dataset].
> Zenodo. https://doi.org/10.5281/zenodo.20929034
> 许可：CC BY 4.0

## 对应论文（Publication）

> Chen, J., Ma, Y., Zhao, B., Wang, R., Chen, G., et al.
> **TCMCPD: An Integrated TCM Case and Pharmacology Database**.
> *Research Square* (preprint), 2026.
> DOI: https://doi.org/10.21203/rs.3.rs-9420486/v1
> Europe PMC ID: PPR1186709

**状态：预印本（preprint），未见同行评审期刊版本。** 写相关工作要标 preprint，不能写成已发表期刊论文。

## 本地已保存

| 文件 | 说明 |
|---|---|
| `paper_TCMCPD_ResearchSquare_rs-9420486_v1.pdf` | 论文全文，22 页，1,442.05 KB<br>sha256 `a8e85f0cd0dc60197919e17d503905bd833054117c38431a73a9fe7e537dac7a`<br>已核验首页标题与作者单位（北京大学 / 北京中医药大学） |

## ⚠️ 论文与数据的两处不一致（引用前必读）

**1）论文的 Data Availability 没有指向这个 Zenodo 记录。**
论文原文写的是：*"Main data and related functions are publicly available through a web interface at http://182.92.96.53:8081/. To obtain the complete data, please contact the corresponding author."*
也就是说作者要求「完整数据请联系通讯作者」，并未声明 Zenodo 那份是完整版。

**2）论文正文的数据统计与本数据集不符。**

| 实体 | 论文正文 | 官方 API（:3000/api） | 本地 Zenodo 版 |
|---|---|---|---|
| 患者 | 5,172 | **4,895** | 4,895 |
| 就诊记录 | 12,900 | 12,900 | 12,900 |
| 症状 | 1,717 | **1,487** | 1,487 |
| 证候 | 233 | 233 | 233 |
| 中药 | 499 | **703** | 703 |
| 成分 | 19,595 | — | 12,640 |
| 靶点 | 5,235 | — | 20,965 |

**官方 API 与 Zenodo 版五项实体计数完全一致**，说明这份 Zenodo 数据就是官方现行发布的完整数据集；**偏大的是论文正文的数字，属统计口径过期**。就诊数（12,900）两边一致，是最关键的时序规模指标，不受影响。

**引用建议**：数据引 Zenodo DOI，方法引 preprint DOI；若要写「数据集规模」，**用本目录实测数字（4,895 患者 / 12,900 就诊），不要抄论文正文的 5,172**。

## 核对依据

1. 论文 PDF 首页标题与 Zenodo 标题一致（正文作 "TCM Case and Pharmacology Database"，Zenodo 作 "TCM Case records and Pharmacology Database"，同一工作）。
2. 作者单位与 Zenodo `creator` 字段一致；年份同为 2026。
3. 用论文给出的官方接口 `http://182.92.96.53:3000/api/{patient,visit,symptom,syndrome,herb}` 读取 `total`，与本地数据逐一比对，五项全中。
4. 注意：Zenodo 记录本身**没有**填 `related_identifiers`，所以「论文↔数据」的对应关系是**按标题+作者+年份+官方接口计数**四重推定，不是作者自报链接。引用时把两个 DOI 都给出来最稳妥。
