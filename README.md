### **應用範例程式碼解析與說明**

此次討論中的應用範例基於 **Spring Boot**、**Milvus** 和 **Spring AI**，旨在實現列車訂票資料的向量化處理與相似度查詢，並包含異常行為檢測的功能。以下是對應程式碼的解析以及其功能的簡要說明：

---

### **1. 自訂嵌入模型 (`CustomEmbeddingModel`)**
此嵌入模型負責將輸入的結構化資料（如用戶 ID、列車 ID 等）轉換為浮點數向量。這些向量將用於後續的相似度檢索和行為分析。

```java
public class CustomEmbeddingModel implements EmbeddingModel {
    @Override
    public EmbeddingResponse call(EmbeddingRequest request) {
        List<String> inputs = request.getInstructions();
        List<Embedding> embeddings = new ArrayList<>();

        for (String input : inputs) {
            float[] embeddingVector = generateEmbedding(input);
            embeddings.add(new Embedding(embeddingVector, 0));
        }

        return new EmbeddingResponse(embeddings);
    }

    private float[] generateEmbedding(String content) {
        char[] chars = content.toCharArray();
        float[] result = new float[chars.length];
        for (int i = 0; i < chars.length; i++) {
            result[i] = (float) chars[i];
        }
        return result;
    }
}
```

- **解析**：`generateEmbedding` 方法將每個字符轉換為對應的浮點數，並將其作為向量的一部分。這是一個簡單的字符轉換示例，實際應用中可以用更複雜的嵌入模型來表示這些文本資料。
- **作用**：此嵌入模型的作用是將結構化資料轉換為數值向量，便於後續的向量操作（如相似度計算）。

---

### **2. 插入數據到 Milvus (`MilvusService`)**
此服務負責將向量化資料插入到 Milvus 資料庫，並構建索引以加速後續的查詢。

```java
public void insertTicket(Ticket ticket) {
    List<String> ticketInputs = Arrays.asList(
        ticket.getUserId(),
        ticket.getTrainId(),
        ticket.getDeparture(),
        ticket.getDestination(),
        ticket.getBookingTime()
    );

    EmbeddingRequest request = new EmbeddingRequest(ticketInputs, null);
    EmbeddingResponse response = embeddingModel.call(request);

    float[] embeddingOutput = response.getResults().get(0).getOutput();
    List<Float> vector = adjustVectorDimensions(embeddingOutput, 128);

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
```

- **解析**：`insertTicket` 方法將列車訂票資料轉換為向量後，將其插入到 Milvus 中，並將關鍵欄位（如 `userId`, `trainId`）作為其他屬性進行存儲。
- **作用**：資料庫內的向量可以用來進行相似度查詢，從而找到與當前訂票行為相似的其他記錄。

---

### **3. 相似度查詢與搜索 (`searchSimilarTickets`)**
此方法用於查詢與輸入資料相似的數據，並返回相關結果。

```java
public List<SearchResultsWrapper.IDScore> searchSimilarTickets(Ticket ticket, int topK) {
    createIndex();
    loadCollection();

    List<String> ticketInputs = List.of(
        ticket.getUserId(),
        ticket.getTrainId(),
        ticket.getDeparture(),
        ticket.getDestination(),
        ticket.getBookingTime()
    );

    EmbeddingRequest request = new EmbeddingRequest(ticketInputs, null);
    EmbeddingResponse response = embeddingModel.call(request);

    float[] embeddingOutput = response.getResults().get(0).getOutput();
    List<Float> queryVector = adjustVectorDimensions(embeddingOutput, 128);

    SearchParam searchParam = SearchParam.newBuilder()
        .withCollectionName("ticket_collection")
        .withMetricType(MetricType.L2)
        .withTopK(topK)
        .withVectors(List.of(queryVector))
        .withVectorFieldName("vector")
        .withOutFields(List.of("userId", "trainId"))
        .build();

    R<SearchResults> searchResponse = milvusClient.search(searchParam);
    if (searchResponse.getStatus() == 0 && searchResponse.getData() != null) {
        SearchResultsWrapper wrapper = new SearchResultsWrapper(searchResponse.getData().getResults());
        return wrapper.getIDScore(0);
    } else {
        System.err.println("搜索失敗: " + searchResponse.getException().getMessage());
        return null;
    }
}
```

- **解析**：此方法根據向量相似度（使用 L2 距離）來搜索與輸入票務資料相似的數據。Milvus 根據向量匹配度返回最接近的結果。
- **作用**：此功能可以用於類似推薦系統，基於過去的行為推薦相似的訂票選擇，或檢測用戶是否存在異常行為。

---

### **4. 測試範例與相似度比對**

- **測試資料**：假設我們插入兩筆資料：
  1. `{"userId": "user1", "trainId": "trainA", "departure": "stationA", "destination": "stationB", "bookingTime": "2024-09-10 12:30"}`
  2. `{"userId": "user2", "trainId": "trainB", "departure": "stationC", "destination": "stationD", "bookingTime": "2024-09-11 14:30"}`

  接著，我們執行如下查詢：
  ```json
  {
    "userId": "user1",
    "trainId": "trainA",
    "departure": "stationA",
    "destination": "stationB",
    "bookingTime": "2024-09-10 12:30"
  }
  ```

- **結果分析**：
  - 查詢結果顯示第一筆資料的相似度較高，`score` 可能接近 0，表示完全匹配。
  - 第二筆資料的相似度會較低，`score` 可能較高（如 1000），表示其與查詢向量的距離較大。

- **測試案例**：
  - **高相似度**：當查詢的票務資料與數據庫中的某筆資料完全匹配（如時間、列車、目的地等相同），相似度 `score` 接近 0，表示非常相似。
  - **低相似度**：當查詢資料與數據庫中的資料有明顯差異（如出發地、列車 ID 完全不同），相似度分數會較高（如數百或數千），表示不相似。

### **總結**

此系統的相似度查詢功能基於向量化資料進行運作，通過 L2 距離度量資料間的相似性。實務上，此功能可以應用於推薦系統、異常行為檢測等場景。在測試中，高度匹配的資料會得到較低的相似度分數，與輸入資料差異大的數據則會得到較高的分數。
