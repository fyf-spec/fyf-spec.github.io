# 第二章：应用层

应用层是网络应用直接所在的层。它不运行在路由器等网络核心设备上，而是运行在端系统中：浏览器、Web 服务器、邮件客户端、DNS 服务器、P2P 节点、视频播放器和 CDN 节点都属于应用层系统。本章的主线是：**网络应用如何组织进程通信，如何选择传输服务，以及典型应用层协议如何设计消息、状态和扩展机制。**

## 目录

- [2.1 网络应用原理](#_2-1-网络应用原理)
- [2.2 应用层协议与传输服务需求](#_2-2-应用层协议与传输服务需求)
- [2.3 Web 与 HTTP](#_2-3-web-与-http)
- [2.4 HTTP 状态、缓存与版本演进](#_2-4-http-状态-缓存与版本演进)
- [2.5 电子邮件：SMTP 与邮件访问](#_2-5-电子邮件-smtp-与邮件访问)
- [2.6 DNS：域名系统](#_2-6-dns-域名系统)
- [2.7 P2P 与 BitTorrent](#_2-7-p2p-与-bittorrent)
- [2.8 视频流与 CDN](#_2-8-视频流与-cdn)
- [2.9 Socket 编程：UDP 与 TCP](#_2-9-socket-编程-udp-与-tcp)
- [2.10 应用层协议速查](#_2-10-应用层协议速查)
- [2.11 本章小结](#_2-11-本章小结)

---

## 2.1 网络应用原理

创建网络应用，就是编写运行在不同端系统上的程序，让这些程序通过网络交换报文。应用代码只需要部署在边缘主机上，网络核心设备不运行用户应用程序，这使得 Web、邮件、视频、P2P、即时通信等应用可以快速演进。

### 客户端-服务器架构

客户端-服务器 (client-server) 是最常见的应用架构。

| 角色 | 特征 |
| --- | --- |
| 服务器 | 始终在线，通常有固定 IP，常部署在数据中心以支撑规模化访问 |
| 客户端 | 主动发起请求，可能间歇在线，可能使用动态 IP，客户端之间通常不直接通信 |

典型例子包括 HTTP、IMAP、FTP。服务器负责等待请求并响应，客户端负责主动连接并获取服务。

### P2P 架构

P2P (peer-to-peer) 不依赖始终在线的中心服务器。任意端系统都可以直接通信，每个 peer 既请求服务，也为其他 peer 提供服务。

| 优点 | 挑战 |
| --- | --- |
| 自扩展性强：新 peer 带来下载需求，也带来上传能力 | 节点动态加入退出，IP 地址变化，管理复杂 |
| 不需要中心服务器承载全部上传压力 | 安全、激励、发现和一致性都更难 |

典型例子是 BitTorrent 文件共享。

### 进程通信与套接字

应用层通信的主体是 **进程**。同一主机上的进程可以通过操作系统 IPC 通信；不同主机上的进程通过网络交换报文。

在应用进程和传输层之间，接口是 **套接字** (socket)。可以把 socket 想成进程通向网络的门：

```text
application process -> socket -> transport protocol -> network
```

开发者控制应用层逻辑，并通过 socket API 选择 TCP 或 UDP；传输层、网络层和链路层的具体传输由操作系统和网络协议栈完成。

### 进程寻址

仅知道主机 IP 地址还不够，因为一台主机可以同时运行多个网络进程。定位应用进程需要：

| 标识 | 作用 |
| --- | --- |
| IP 地址 | 定位主机 |
| 端口号 | 定位主机上的具体进程 |

常见端口：

| 服务 | 端口 | 传输协议 |
| --- | --- | --- |
| HTTP | 80 | TCP |
| HTTPS | 443 | TCP |
| SMTP | 25 | TCP |
| DNS | 53 | UDP / TCP |
| FTP 控制连接 | 21 | TCP |
| SSH | 22 | TCP |

## 2.2 应用层协议与传输服务需求

应用层协议规定进程之间如何交换报文。一个协议通常定义：

| 内容 | 含义 |
| --- | --- |
| 报文类型 | 请求、响应、控制报文等 |
| 报文语法 | 字段如何排列、如何分隔 |
| 报文语义 | 字段值代表什么 |
| 交互规则 | 什么时候发送、如何响应、错误如何处理 |

协议可以是开放协议，例如 HTTP、SMTP、DNS；也可以是专有协议，例如某些视频会议或即时通信协议。

### 应用对传输服务的需求

不同应用对传输层服务要求不同：

| 需求 | 强需求应用 | 可容忍应用 |
| --- | --- | --- |
| 数据完整性 | 文件传输、邮件、网页、金融交易 | 实时音视频可容忍少量丢失 |
| 吞吐量 | 视频流、云同步、大文件传输 | 文本消息、普通网页较弹性 |
| 时延 | 互动游戏、网络电话、视频会议 | 邮件、文件下载 |
| 安全性 | 登录、支付、邮件、现代 Web | 几乎所有现代应用都倾向需要 |

### TCP、UDP 与安全

| 协议 | 提供 | 不提供 |
| --- | --- | --- |
| TCP | 可靠传输、流量控制、拥塞控制、连接建立 | 时延保证、最低吞吐保证、内置加密 |
| UDP | 进程到进程的无连接数据报服务 | 可靠性、流量控制、拥塞控制、连接建立、安全性 |

普通 TCP 和 UDP 都是明文传输。安全通常由 TLS 提供。TLS 位于应用层和 TCP 之间，为应用提供加密、完整性和认证，例如 HTTPS。

## 2.3 Web 与 HTTP

Web 页面由对象组成。对象可以是 HTML 文件、图片、CSS、JavaScript、视频片段等，每个对象由 URL 标识。HTTP (HyperText Transfer Protocol) 是 Web 的核心应用层协议。

HTTP 使用客户端-服务器模型：

| 角色 | 行为 |
| --- | --- |
| HTTP 客户端 | 浏览器发起 TCP 连接，发送请求，接收并展示对象 |
| HTTP 服务器 | Web 服务器等待连接，解析请求，返回响应对象 |

HTTP 通常使用 TCP。客户端连接服务器 80 端口，双方在 TCP 连接上传输 HTTP 请求和响应。HTTP 协议本身是 **无状态** 的：服务器不会因为协议本身而记住客户端过去的请求。

### 非持久 HTTP

非持久 HTTP 中，每个对象使用一个单独的 TCP 连接：

1. 客户端向服务器建立 TCP 连接。
2. 客户端发送 HTTP 请求。
3. 服务器返回对象。
4. 服务器关闭 TCP 连接。

若忽略文件传输时间，每个对象大约需要：

$$
2RTT
$$

更完整地说：

$$
\text{response time} = 2RTT + \text{file transmission time}
$$

其中一个 RTT 用于 TCP 建立连接，另一个 RTT 用于请求到达和响应首字节返回。

### 持久 HTTP

HTTP/1.1 默认使用持久连接：服务器发送响应后保持 TCP 连接，后续对象可复用同一连接。

| 方式 | 特点 |
| --- | --- |
| 非持久连接 | 每个对象一个 TCP 连接，开销大 |
| 持久连接 | 多个对象复用连接，减少 RTT 和操作系统开销 |
| 流水线 | 客户端可连续发送请求，不必等待前一个响应返回 |

### HTTP 请求报文

HTTP 请求报文是 ASCII 文本，典型格式如下：

```http
GET /index.html HTTP/1.1
Host: gaia.cs.umass.edu
User-Agent: Mozilla/5.0
Accept: text/html
```

常见方法：

| 方法 | 作用 |
| --- | --- |
| GET | 请求对象，也可在 URL 的 `?` 后携带参数 |
| POST | 在实体主体中向服务器提交数据 |
| HEAD | 类似 GET，但只返回响应头 |
| PUT | 上传对象到服务器指定路径 |

### HTTP 响应报文

响应报文包含状态行、首部行和可选实体主体：

```http
HTTP/1.1 200 OK
Date: ...
Server: ...
Content-Type: text/html
Content-Length: ...

<html>...</html>
```

常见状态码：

| 状态码 | 含义 |
| --- | --- |
| 200 OK | 请求成功 |
| 301 Moved Permanently | 对象永久移动 |
| 304 Not Modified | 缓存副本仍有效，无需返回对象 |
| 400 Bad Request | 请求格式错误 |
| 404 Not Found | 对象不存在 |
| 500 Internal Server Error | 服务器内部错误 |
| 505 HTTP Version Not Supported | 不支持该 HTTP 版本 |

## 2.4 HTTP 状态、缓存与版本演进

HTTP 无状态可以简化服务器设计，但真实 Web 应用需要登录、购物车、个性化推荐等状态。HTTP 通过 Cookie 和服务器端数据库组合实现状态管理。

### Cookie

Cookie 机制包含四个部分：

| 部分 | 作用 |
| --- | --- |
| 响应中的 `Set-Cookie` 首部 | 服务器给客户端设置标识 |
| 请求中的 `Cookie` 首部 | 浏览器后续请求自动带上标识 |
| 浏览器本地 Cookie 文件 | 保存站点相关标识 |
| 服务器后端数据库 | 将 Cookie 标识映射到用户状态 |

Cookie 可用于会话状态、购物车、个性化推荐和登录保持，也可能被第一方或第三方用于跟踪用户行为，因此带来隐私问题。

### Web 缓存

Web 缓存，也叫代理服务器，目标是在不访问源服务器的情况下满足客户端请求。

工作流程：

1. 浏览器把 HTTP 请求发给缓存。
2. 若缓存命中，缓存直接返回对象。
3. 若未命中，缓存向源服务器请求对象。
4. 缓存保存对象副本，再返回给客户端。

Web 缓存的价值：

- 降低客户端响应时间，因为缓存更靠近用户。
- 减少机构接入链路流量。
- 降低源服务器压力。
- 让资源较少的内容提供者也能被缓存基础设施加速。

课件示例中，若缓存命中率为 0.4，则只有 60% 请求需要经过源服务器，接入链路利用率和排队时延都会显著下降。

### 条件 GET

条件 GET 避免缓存对象过期后盲目重传整个对象。

客户端发送：

```http
GET /object HTTP/1.1
If-Modified-Since: <date>
```

服务器行为：

| 情况 | 响应 |
| --- | --- |
| 对象未修改 | `304 Not Modified`，不发送对象主体 |
| 对象已修改 | `200 OK`，发送新对象 |

### HTTP/2 与 HTTP/3

HTTP/2 的目标是降低多对象页面加载延迟。它保留方法、状态码和大多数首部语义，但引入：

- 在单个 TCP 连接上复用多个对象流。
- 将对象分割成帧，交错传输。
- 根据优先级调度对象。
- 服务器推送未显式请求但可能需要的对象。

HTTP/2 缓解了 HTTP/1.1 按请求顺序响应造成的应用层 HOL 阻塞，但由于仍运行在单个 TCP 连接上，TCP 层丢包仍会阻塞后续字节交付。

HTTP/3 基于 QUIC over UDP，在 UDP 上实现安全、可靠性、拥塞控制和对象级别的流复用，进一步减少 TCP 队头阻塞问题。

## 2.5 电子邮件：SMTP 与邮件访问

电子邮件系统由三部分组成：

| 组成 | 作用 |
| --- | --- |
| 用户代理 UA | 撰写、编辑、阅读邮件，如 Outlook、手机邮件客户端 |
| 邮件服务器 | 保存用户邮箱，维护待发送邮件队列 |
| SMTP | 邮件服务器之间传输邮件的协议 |

### 邮件发送流程

Alice 给 Bob 发邮件的过程：

1. Alice 用用户代理撰写邮件。
2. Alice 的用户代理把邮件提交到 Alice 的邮件服务器。
3. Alice 邮件服务器中的 SMTP 客户端连接 Bob 邮件服务器。
4. SMTP 通过 TCP 连接把邮件传给 Bob 邮件服务器。
5. Bob 邮件服务器把邮件放入 Bob 的邮箱。
6. Bob 使用用户代理通过邮件访问协议读取邮件。

SMTP 使用 TCP 25 端口，采用命令/响应交互，命令和响应是 ASCII 文本。

### SMTP 交互

典型交互：

```text
S: 220 hamburger.edu
C: HELO crepes.fr
S: 250 Hello crepes.fr
C: MAIL FROM:<alice@crepes.fr>
S: 250 Sender ok
C: RCPT TO:<bob@hamburger.edu>
S: 250 Recipient ok
C: DATA
S: 354 End with "." on a line by itself
C: Do you like ketchup?
C: .
S: 250 Message accepted for delivery
C: QUIT
S: 221 closing connection
```

SMTP 的三个阶段：

| 阶段 | 内容 |
| --- | --- |
| 握手 | `HELO` / `EHLO` 等问候 |
| 消息传输 | `MAIL FROM`、`RCPT TO`、`DATA` |
| 关闭 | `QUIT` |

SMTP 与 HTTP 对比：

| 维度 | HTTP | SMTP |
| --- | --- | --- |
| 方向 | 客户端拉取对象 | 发送方推送邮件 |
| 连接 | TCP | TCP |
| 交互 | ASCII 请求/响应和状态码 | ASCII 命令/响应和状态码 |
| 对象组织 | 每个对象在独立响应中返回 | 多个对象可放在 multipart 邮件中 |
| 限制 | Web 对象格式灵活 | 传统 SMTP 要求 7-bit ASCII |

### 邮件消息格式与访问协议

SMTP 定义邮件如何传输；邮件消息本身的格式由邮件格式 RFC 定义，包含首部和正文。

常见首部：

```text
To:
From:
Subject:
```

注意这些邮件首部不同于 SMTP 命令中的 `MAIL FROM` 和 `RCPT TO`。

邮件访问协议用于从接收方邮件服务器读取邮件：

| 协议 | 作用 |
| --- | --- |
| IMAP | 邮件保存在服务器上，支持检索、删除、文件夹管理和多端同步 |
| POP3 | 简单拉取邮件，常用于下载后本地管理 |
| HTTP | Gmail、Outlook Web 等 Webmail 界面常用 HTTP/HTTPS 访问 |

## 2.6 DNS：域名系统

DNS (Domain Name System) 把人类可读的主机名映射到 IP 地址。它是一个分布式、层次化数据库，也是一个应用层协议。

DNS 提供的服务：

| 服务 | 说明 |
| --- | --- |
| 主机名到 IP 地址转换 | `www.example.com -> IP` |
| 主机别名 | CNAME 将别名映射到规范名 |
| 邮件服务器别名 | MX 记录指定邮件服务器 |
| 负载分配 | 一个名称可对应多个 IP 地址 |

DNS 不能集中化，因为集中式 DNS 会有单点故障、流量瓶颈、远距离访问和维护困难，无法支撑互联网规模。

### 层次结构

DNS 层次包括：

| 层次 | 作用 |
| --- | --- |
| 根 DNS 服务器 | 当本地服务器不知道去哪里时给出顶级域线索 |
| TLD 服务器 | 管理 `.com`、`.org`、`.edu`、`.cn` 等顶级域 |
| 权威 DNS 服务器 | 组织自己的 DNS 服务器，给出该组织主机的权威记录 |
| 本地 DNS 服务器 | 用户查询第一站，通常由 ISP、学校或企业提供，负责缓存和递归/迭代解析 |

本地 DNS 服务器不严格属于层次结构，但它是主机查询 DNS 的第一跳。

### 迭代查询与递归查询

迭代查询中，被查询服务器只返回下一步应该联系谁：

```text
client -> local DNS -> root -> TLD -> authoritative
```

每一层返回线索，由本地 DNS 继续向下问。

递归查询中，被查询服务器代替请求者继续查询，直到拿到最终结果再返回。递归把解析负担放到被查询服务器上，因此高层 DNS 通常不愿承担大规模递归负载。

### DNS 缓存

任何 DNS 服务器学到映射后都可以缓存。缓存能降低解析时延和上层服务器负载。每条记录都有 TTL，TTL 到期后缓存失效。

缓存的代价是：如果主机 IP 改变，在相关缓存 TTL 到期前，互联网中仍可能有人拿到旧映射。因此 DNS 是一种尽力而为的名称解析系统。

### DNS 资源记录

DNS 数据库存储资源记录 (Resource Record, RR)：

```text
(name, value, type, ttl)
```

常见类型：

| Type | name | value | 含义 |
| --- | --- | --- | --- |
| A | 主机名 | IP 地址 | 主机名到 IPv4 地址 |
| NS | 域名 | 权威 DNS 服务器主机名 | 指定该域由谁负责 |
| CNAME | 别名 | 规范主机名 | 别名到真实名称 |
| MX | 域名 | 邮件服务器主机名 | 指定该域邮件服务器 |

### DNS 报文

DNS 查询和响应格式相同，首部包含：

| 字段 | 作用 |
| --- | --- |
| identification | 16 bit 查询 ID，响应使用相同 ID 以便匹配 |
| flags | 查询/响应、递归期望、递归可用、权威回答等标志 |
| # questions | 查询问题数量 |
| # answer RRs | 回答资源记录数量 |
| # authority RRs | 权威区记录数量 |
| # additional RRs | 额外记录数量 |

报文后面依次是 questions、answers、authority 和 additional info。

### DNS 安全

DNS 面临：

- DDoS 攻击根服务器或 TLD 服务器。
- DNS 查询拦截和虚假响应。
- DNS 缓存投毒。

DNSSEC 用于提供认证和消息完整性，缓解伪造响应和缓存投毒问题。

## 2.7 P2P 与 BitTorrent

P2P 的核心优势是自扩展：更多 peer 不仅带来更多下载需求，也带来更多上传能力。

### 文件分发时间

设文件大小为 $F$，服务器上传速率为 $u_s$，第 $i$ 个 peer 的下载速率为 $d_i$，上传速率为 $u_i$，最小下载速率为 $d_{min}$。

客户端-服务器架构中：

```text
服务器必须上传 N 份副本，每个客户端必须下载 1 份
```

分发时间满足：

$$
D_{CS} \ge \max\left\{\frac{NF}{u_s}, \frac{F}{d_{min}}\right\}
$$

P2P 架构中：

- 服务器至少上传一份。
- 每个 peer 至少下载一份。
- 所有 peer 合计要下载 $NF$ bit。
- 总上传能力是 $u_s + \sum_i u_i$。

分发时间满足：

$$
D_{P2P} \ge \max\left\{\frac{F}{u_s}, \frac{F}{d_{min}}, \frac{NF}{u_s + \sum_i u_i}\right\}
$$

随着 $N$ 增长，客户端-服务器需要服务器上传更多副本；P2P 则同时增加系统上传能力。

### BitTorrent

BitTorrent 把文件切成块，课件中块大小为 256 KB。参与同一个文件交换的一组 peer 称为 torrent，tracker 负责跟踪参与该 torrent 的 peer 列表。

加入过程：

1. Alice 加入 torrent。
2. Alice 向 tracker 注册并获取部分 peer 列表。
3. Alice 连接这些 peer，成为邻居。
4. Alice 从邻居处获取块，同时也向邻居上传自己已有的块。
5. peer 可能动态加入、退出，即 churn。

请求块策略：

| 策略 | 作用 |
| --- | --- |
| 最稀缺优先 rarest first | 优先请求副本少的块，避免某些块消失 |
| 块列表交换 | peer 周期性告诉邻居自己拥有哪些块 |

上传策略是 tit-for-tat：

- Alice 优先向当前给她最高下载速率的 4 个 peer 上传。
- 每 10 秒重新评估前 4 名。
- 每 30 秒随机选择一个额外 peer 乐观解阻塞。

这个机制既鼓励贡献上传，也给新 peer 机会找到更好的交换伙伴。

## 2.8 视频流与 CDN

视频流是互联网带宽主要消耗者之一。挑战来自规模和异质性：用户数量巨大，接入带宽、终端能力和网络拥塞状况差异很大。

### 视频编码与播放约束

视频是一系列图像按固定速率播放。编码通过空间冗余和时间冗余减少比特数：

| 冗余 | 例子 |
| --- | --- |
| 空间冗余 | 同一帧中大块相同颜色不重复发送每个像素 |
| 时间冗余 | 相邻帧相似，只发送差异 |

视频编码可以是：

| 类型 | 含义 |
| --- | --- |
| CBR | constant bit rate，固定码率 |
| VBR | variable bit rate，根据内容复杂度改变码率 |

流式存储视频的主要问题：

- 服务器到客户端带宽随时间波动。
- 拥塞导致丢包和延迟。
- 播放需要连续进行，因此客户端需要缓冲。
- 用户还需要暂停、快进、跳过等交互。

### DASH

DASH (Dynamic Adaptive Streaming over HTTP) 的思想是：

1. 服务器把视频切成多个小块。
2. 每个块以多个码率版本编码。
3. 服务器提供 manifest 文件，列出不同块和码率对应的 URL。
4. 客户端周期性估计可用带宽。
5. 客户端自行决定下一个块请求哪个码率、何时请求、向哪个服务器请求。

客户端智能是 DASH 的关键：带宽高时请求高码率块，带宽低时切到低码率，以尽量避免卡顿。

### CDN

CDN (Content Distribution Network) 把内容副本部署到离用户更近的位置。这样可以减少跨网络长路径传输，降低时延和源站压力。

两种部署思路：

| 类型 | 说明 |
| --- | --- |
| 深入接入网络 | CDN 服务器部署进大量 ISP 接入网络，靠近用户 |
| 大型集群 | 在少数关键位置部署大型数据中心集群 |

CDN 要回答三个问题：

1. 哪些内容放在哪些 CDN 节点。
2. 用户应该访问哪个 CDN 节点。
3. 以什么码率或方式向用户传输。

### Netflix 案例

Netflix 使用混合架构：

| 部分 | 作用 |
| --- | --- |
| AWS 控制平面 | 用户注册、账号、推荐、搜索等控制逻辑 |
| OpenConnect CDN | 存储视频副本并负责向用户传输视频 |
| DASH 客户端 | 根据带宽和缓冲状态选择视频块码率 |

用户请求视频时，Netflix 通过 DNS / manifest / CDN 选择机制把用户导向合适的 OpenConnect 节点，再由客户端按 DASH 方式请求视频块。

## 2.9 Socket 编程：UDP 与 TCP

socket 是应用进程与端到端传输协议之间的门。UDP 和 TCP 提供两类不同 socket。

| 传输服务 | Socket 类型 | 特点 |
| --- | --- | --- |
| UDP | `SOCK_DGRAM` | 无连接，不可靠数据报，每次发送要指定目的地址 |
| TCP | `SOCK_STREAM` | 面向连接，可靠字节流，连接建立后不必每次指定目的地址 |

### UDP socket 交互

UDP 服务器：

1. 创建 UDP socket。
2. 绑定本地端口。
3. 使用 `recvfrom()` 接收数据和客户端地址。
4. 使用 `sendto()` 回复该地址。

UDP 客户端：

1. 创建 UDP socket。
2. 用 `sendto()` 把数据和服务器地址一起交给 socket。
3. 用 `recvfrom()` 接收回复。

```python
from socket import *

clientSocket = socket(AF_INET, SOCK_DGRAM)
clientSocket.sendto("hello".encode(), ("localhost", 12000))
message, serverAddress = clientSocket.recvfrom(2048)
clientSocket.close()
```

### TCP socket 交互

TCP 服务器有两类 socket：

| Socket | 作用 |
| --- | --- |
| welcome socket | 监听端口，等待新连接 |
| connection socket | `accept()` 后创建，专门服务某个客户端 |

TCP 客户端：

1. 创建 TCP socket。
2. 调用 `connect()`，触发 TCP 三次握手。
3. 连接建立后用 `send()` / `recv()` 交换字节流。

TCP 服务器：

1. 创建 welcome socket。
2. `bind()` 到本地端口。
3. `listen()` 开始监听。
4. `accept()` 阻塞等待连接，返回 connection socket。
5. 用 connection socket 与该客户端通信。

```python
from socket import *

serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(("", 12000))
serverSocket.listen(1)

while True:
    connectionSocket, addr = serverSocket.accept()
    sentence = connectionSocket.recv(1024).decode()
    connectionSocket.send(sentence.upper().encode())
    connectionSocket.close()
```

## 2.10 应用层协议速查

| 协议 | 默认端口 | 传输层 | 核心特征 |
| --- | --- | --- | --- |
| HTTP | 80 | TCP | Web 请求/响应，无状态 |
| HTTPS | 443 | TCP + TLS | 加密 Web |
| SMTP | 25 | TCP | 邮件服务器之间推送邮件 |
| IMAP | 143 | TCP | 邮件读取和服务器端文件夹管理 |
| POP3 | 110 | TCP | 简单邮件拉取 |
| DNS | 53 | UDP / TCP | 域名解析，分布式层次数据库 |
| FTP | 20 / 21 | TCP | 文件传输，控制和数据连接分离 |
| SSH | 22 | TCP | 加密远程登录 |

## 2.11 本章小结

本章的核心问题是：**网络应用如何通过端系统上的进程、协议和基础设施完成通信？**

需要掌握的主线：

1. 网络应用运行在端系统上，网络核心不运行用户应用代码。
2. 应用架构主要包括客户端-服务器和 P2P。
3. 进程通过 socket 使用传输层服务，IP 地址定位主机，端口号定位进程。
4. 应用层协议定义报文类型、语法、语义和交互规则。
5. HTTP 是无状态请求/响应协议，持久连接、Cookie、缓存、条件 GET、HTTP/2/3 都是围绕性能和状态管理展开的扩展。
6. SMTP 负责邮件服务器之间推送邮件，IMAP/POP/HTTP 负责用户读取邮件。
7. DNS 是分布式层次化数据库，用资源记录完成名称解析、别名、邮件服务器定位和负载分配。
8. P2P 通过 peer 上传能力实现自扩展，BitTorrent 用稀缺优先和 tit-for-tat 提高效率和激励上传。
9. 视频流依赖编码、缓冲、DASH 自适应码率和 CDN 边缘分发。
10. UDP socket 面向数据报，TCP socket 面向可靠字节流，并使用 welcome socket 与 connection socket 分离监听和通信。

下一章进入传输层：应用层把消息交给 socket 后，TCP 和 UDP 如何在端系统之间提供进程到进程的通信。
