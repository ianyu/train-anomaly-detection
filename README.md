### **一、技術架構與引用套件**

本應用主要使用以下技術：
- **Spring Boot**：負責構建 Web API，處理數據傳輸和業務邏輯。
- **Milvus**：高效處理高維度向量數據的數據庫，支持向量相似度檢索。
- **Spring AI**：整合嵌入模型，將結構化或非結構化數據轉換為固定維度的向量。
- **自定義嵌入模型**：將訂票資料（如用戶 ID、列車 ID、出發地、目的地、訂票時間）轉換為向量，便於後續相似查詢。

#### Maven 引用套件：
```xml
<dependencies>
    <!-- Spring Boot Web -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>

    <!-- Milvus SDK for Java -->
    <dependency>
        <groupId>io.milvus</groupId>
        <artifactId>milvus-sdk-java</artifactId>
        <version>2.4.3</version>
    </dependency>

    <!-- Spring AI for embedding model -->
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-milvus-store-spring-boot-starter</artifactId>
    </dependency>

    <!-- Jakarta Annotation for @PostConstruct and @PreDestroy -->
    <dependency>
        <groupId>jakarta.annotation</groupId>
        <artifactId>jakarta.annotation-api</artifactId>
        <version>2.1.1</version>
    </dependency>
</dependencies>
```

#### Milvus 安裝與設定：
```bash
# 使用 Docker 快速安裝 Milvus
docker pull milvusdb/milvus:v2.4.3
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:v2.4.3
```

---

### **二、數據流與系統流程**

#### 1. **資料預處理與向量化**
系統接收到列車訂票資料的 JSON 格式輸入後，會將其通過自定義嵌入模型轉換為向量（例如 128 維）。這些向量代表了用戶訂票行為的語義特徵，用於相似度查詢和異常檢測。

- **資料轉向量的主要邏輯**：
    ```java
    public float[] convertJsonToVector(Ticket ticket) {
        String combinedText = String.join(" ",
            ticket.getUserId(),
            ticket.getTrainId(),
            ticket.getDeparture(),
            ticket.getDestination(),
            ticket.getBookingTime()
        );

        EmbeddingRequest request = EmbeddingRequest.builder()
            .withInputs(List.of(combinedText))
            .build();

        EmbeddingResponse response = embeddingModel.call(request);
        return adjustVectorDimensions(response.getResults().get(0).getOutput(), 128);
    }
    ```

#### 2. **向量維度調整**
嵌入模型可能生成不同維度的向量，但系統要求所有向量必須為 128 維。當向量不足 128 維時，會進行補零操作；超過時，則進行截斷。

- **向量維度調整邏輯**：
    ```java
    public List<Float> adjustVectorDimensions(float[] embeddingOutput, int requiredDimension) {
        List<Float> vector = floatArrayToList(embeddingOutput);
        while (vector.size() < requiredDimension) {
            vector.add(0.0f);  // 使用 0 來填充
        }
        return vector.subList(0, requiredDimension);
    }
    ```

#### 3. **數據插入與索引建構**
Milvus 用於存儲這些向量化的數據，並通過索引加速查詢。索引方式使用 `IVF_FLAT`，度量方法選用 `L2`（歐氏距離）。

- **索引構建與數據插入**：
    ```java
    public void insertTicket(Ticket ticket) {
        List<Float> vector = convertJsonToVector(ticket);
        List<InsertParam.Field> fields = Arrays.asList(
            new InsertParam.Field("userId", Arrays.asList(ticket.getUserId())),
            new InsertParam.Field("trainId", Arrays.asList(ticket.getTrainId())),
            new InsertParam.Field("vector", List.of(vector))
        );
        InsertParam insertParam = InsertParam.newBuilder()
            .withCollectionName("ticket_collection")
            .withFields(fields)
            .build();
        milvusClient.insert(insertParam);
    }

    public void createIndex() {
        milvusClient.createIndex(CreateIndexParam.newBuilder()
            .withCollectionName("ticket_collection")
            .withFieldName("vector")
            .withIndexType(IndexType.IVF_FLAT)
            .withMetricType(MetricType.L2)
            .build());
    }
    ```

---

### **三、推薦系統與行為分析**

#### 1. **推薦系統**
系統基於用戶的歷史訂票行為進行推薦，通過向量相似度計算來推薦相似的列車行程。

- **推薦邏輯**：
    ```java
    public List<Ticket> recommendTicketsForUser(Ticket currentTicket, int topK) {
        float[] queryVector = convertJsonToVector(currentTicket);
        SearchParam searchParam = SearchParam.newBuilder()
            .withCollectionName("ticket_collection")
            .withTopK(topK)
            .withVectors(List.of(queryVector))
            .build();
        R<SearchResults> searchResponse = milvusClient.search(searchParam);
        // 根據結果進行推薦
    }
    ```

#### 2. **行為分析**
行為分析功能可以檢測異常的訂票行為。例如，當相似度分數超過預設閾值時，標記為異常行為。

- **異常檢測邏輯**：
    ```java
    public boolean isAnomalousBehavior(Ticket ticket) {
        float[] queryVector = convertJsonToVector(ticket);
        SearchParam searchParam = SearchParam.newBuilder()
            .withCollectionName("ticket_collection")
            .withTopK(5)
            .withVectors(List.of(queryVector))
            .build();
        R<SearchResults> searchResponse = milvusClient.search(searchParam);
        if (searchResponse.getStatus() == 0) {
            List<SearchResultsWrapper.IDScore> results = new SearchResultsWrapper(searchResponse.getData().getResults()).getIDScore(0);
            return results.stream().anyMatch(score -> score.getScore() > 1000);
        }
        return true;  // 查詢失敗時，視為異常
    }
    ```

---

### **四、應用場景與未來優化**

#### 1. **應用場景**
- **推薦系統**：基於向量化的用戶行為進行推薦，適用於電商、影音內容、社交媒體等場景。
- **異常行為檢測**：使用向量化資料檢測用戶的異常行為，防止欺詐或異常訂票。

#### 2. **未來優化**
- **模型優化**：可以集成更複雜的嵌入模型（如 BERT、GPT）來提升向量化的效果。
- **索引優化**：根據數據量大小和查詢頻率，可以使用 `HNSW` 等更高效的索引方式來提高查詢性能。

---
