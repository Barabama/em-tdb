
# tdb格式说明

## 基本规则

- 行长度限制 : 每行不超过 80 个字符(源于 Fortran 编写的历史原因)
- 注释符 : 使用 $ 表示注释 软件会忽略 $ 开头的行
  注释可用于添加说明或分隔不同部分以提高可读性

## 1.ELEMENT

定义系统中的元素或组分

```plaintext
ELEMENT <NAME> <ERF_STATE> <ATOMIC_MASS> <H298-H0> <S298>
```

- `<NAME>`: 元素名称
- `<REF_STATE>`: 参考状态(例如晶体结构)
- `<ATOMIC_MASS>`: 原子质量
- `<H298-H0>`: 标准生成焓(单位：J/mol)
- `<S298>`: 标准熵(单位：J/(mol·K))

示例:
ELEMENT /- ELECTRON_GAS 0.0000E+00 0.0000E+00 0.0000E+00 !
ELEMENT VA VACUUM       0.0000E+00 0.0000E+00 0.0000E+00 !
ELEMENT CU FCC_A1       6.3546E+01 5.0041E+03 3.3150E+01 !
ELEMENT MG HCP_A3       2.4305E+01 4.9980E+03 3.2671E+01 !

## 2. FUNCTIONS

定义热力学函数，用于描述纯物质或混合物的 Gibbs 自由能。

```plaintext
FUNCTION <NAME> <TEMP_RANGE> <EXPERESSION>; <TEMP_RANGE> Y/N !
```

- `<NAME>`: 函数名称
- `<TEMP_RANGE>`: 温度范围(单位: K)
- `<EXPERESSION>`: 函数表达式
- `Y/N`: 是否有后续温度区间
- `!`: 结束标志

示例:

```plaintext
FUNCTION GHSERCU 298.15 -7770.458+130.485235*T-24.112392*T*LN(T)
    -.00265684*T**2+1.29223E-07*T**3+52478*T**(-1); 1357.77 Y 
    -13542.026+183.803828*T-31.38*T*LN(T)+3.64167E+29*T**(-9); 3200 N !
FUNCTION GHSERMG 298.15 -8367.34+143.675547*T-26.1849782*T*LN(T)
    +4.858E-04*T**2-1.393669E-06*T**3+78950*T**(-1); 923 Y
    -14130.185+204.716215*T-34.3088*T*LN(T)+1.038192E+28*T**(-9); 3000 N !
FUNCTION GHCPCU 298.15 +GHSERCU+600+0.2*T; 3200 N !
FUNCTION GFCCMG 298.15 +GHSERMG+2600-0.9*T; 3000 N !
FUNCTION GLIQCU 298.15 +GHSERCU+12964.735-9.511904*T-5.8489E-21*T**7; 1357.77 Y
    -46.545+173.881484*T-31.38*T*LN(T); 3200 N !
FUNCTION GLIQMG 298.15 +GHSERMG+8202.243-8.83693*T-8.0176E-20*T**7; 923 Y
    -5439.869+195.324057*T-34.3088*T*LN(T); 3000 N !
```

## 3. PHASE CONSTITUENT

PHASE定义相及相关参数, CONSTITUENT定义各子晶格中的组分

```plaintext
PHASE <NAME> % <SUB_LATTICES> <STOICHIOMETRY>
CONSTITUENT <PHASE> : <COMPONENTS> : !
```

- `<NAME>`: 相名称
- `%`: 固定标识符
- `<SUB_LATTICES>`: 子晶格数量
- `<STOICHIOMETRY>`: 各子晶格的化学计量比
- `<PHASE>`: 相名称
- `:`: 固定标识符
- `<COMPONENTS>`: 组成列表, 用逗号分隔, % 用于标识该子晶格主要成分

示例:

```plaintext
PHASE LIQUID % 1 1.0 !
CONSTITUENT LIQUID :CU,MG : !

PHASE HCP_A3 % 2 1.0 0.5 !
CONSTITUENT HCP_A3 :CU,MG% : VA : !
```

## 4. PARAMETER

定义相的具体热力学参数

```plaintext
PARAMETER <TYPE>(<PHASE>,<COMPONENTS>;<ORDER>) <TYPE_RANGE> <EXPRESSION>; <TEMP_RANGE> Y/N !
```

- `<TYPE>`: 参数类型(如 `G` 表示纯物质，`L` 表示混合参数)
- `<PHASE>`: 相名称
- `<COMPONENTS>`: 成分列表
- `<ORDER>`: Redlich-Kister 展开阶数
- `<TEMP_RANGE>`: 温度范围
- `<EXPRESSION>`: 数学表达式

示例:

```plaintext
PARAMETER G(LIQUID,CU;0) 298.15+GLIQCU; 3200 N!
PARAMETER L(LIQUID,CU,MG;0) 298.15 LIQ_AA+LIQ_AAT*T; 3000 N!
```

## 5. OPTIMIZATION

定义优化变量及其边界条件

```plaintext
OPTIMIZATION <VAR_NAME> <LOWER_BOUND> <START_VALUE> <UPPER_BOUND>; Y/N!
```

- `<VAR_NAME>`: 变量名称
- `<LOWER_BOUND>`: 下界
- `<START_VALUE>`: 初始值
- `<UPPER_BOUND>`: 上界

示例:

```plaintext
OPTIMIZATION LIQ_AA -50000 -34000; 0 N!
OPTIMIZATION CUMG2_H -30000 -27000; 0 N!
```

## 文件结构建议

尽管 TDB 文件是自由格式，但通常遵循以下顺序：

1. **ELEMENT**: 定义元素
2. **FUNCTION**: 定义热力学函数
3. **PHASE** 和 **CONSTITUENT**: 定义相及组成
4. **PARAMETER**: 定义热力学参数
5. **OPTIMIZATION**: 定义优化变量

## 注意事项

1. **文件风格**: 不同用户可能有不同的排版习惯, 共享文件时需注意统一格式
2. **扩展功能**: TDB 文件还可包含磁性贡献、压力依赖性等高级模型, 但这些内容不在当前讨论范围内
