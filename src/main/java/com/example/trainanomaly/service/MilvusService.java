package com.example.trainanomaly.service;

import com.example.trainanomaly.model.Ticket;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.RpcStatus;
import io.milvus.param.index.CreateIndexParam;
import io.milvus.param.collection.LoadCollectionParam;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.grpc.SearchResults;
import io.milvus.grpc.MutationResult;
import io.milvus.client.MilvusServiceClient;
import io.milvus.response.SearchResultsWrapper;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

@Service
public class MilvusService {

    private MilvusServiceClient milvusClient;

    @Autowired
    private EmbeddingModel embeddingModel;

    public MilvusService(MilvusServiceClient milvusClient) {
        this.milvusClient = milvusClient;
    }

    // 1. 插入數據
    public void insertTicket(Ticket ticket) {
        List<String> ticketInputs = Arrays.asList(
                ticket.getUserId(),
                ticket.getTrainId(),
                ticket.getDeparture(),
                ticket.getDestination(),
                ticket.getBookingTime()
        );

        // 呼叫嵌入模型生成向量資料
        EmbeddingRequest request = new EmbeddingRequest(ticketInputs, null);
        EmbeddingResponse response = embeddingModel.call(request);

        // 將 float[] 轉換為 List<Float>
        float[] embeddingOutput = response.getResults().get(0).getOutput();
        List<Float> vector = adjustVectorDimensions(embeddingOutput, 128);

        // 構造 InsertParam 的欄位
        List<InsertParam.Field> fields = Arrays.asList(
                new InsertParam.Field("userId", Arrays.asList(ticket.getUserId())),
                new InsertParam.Field("trainId", Arrays.asList(ticket.getTrainId())),
                new InsertParam.Field("departure", Arrays.asList(ticket.getDeparture())),
                new InsertParam.Field("destination", Arrays.asList(ticket.getDestination())),
                new InsertParam.Field("bookingTime", Arrays.asList(ticket.getBookingTime())),
                new InsertParam.Field("vector", List.of(vector))  // 向量欄位
        );

        // 插入數據
        InsertParam insertParam = InsertParam.newBuilder()
                .withCollectionName("ticket_collection")
                .withFields(fields)
                .build();

        R<MutationResult> resultWrapper = milvusClient.insert(insertParam);
        if (resultWrapper.getStatus() == 0) {
            MutationResult result = resultWrapper.getData();
            System.out.println("成功插入了 " + result.getInsertCnt() + " 筆數據。");
        } else {
            System.err.println("插入失敗: " + resultWrapper.getException().getMessage());
        }
    }

    // 2. 構建索引
    public void createIndex() {
        R<RpcStatus> createIndexResponse = milvusClient.createIndex(
                CreateIndexParam.newBuilder()
                        .withCollectionName("ticket_collection")
                        .withFieldName("vector")
                        .withIndexType(IndexType.IVF_FLAT)
                        .withMetricType(MetricType.L2)
                        .withExtraParam("{\"nlist\": 1024}")
                        .build()
        );

        if (createIndexResponse.getStatus() != 0) {
            System.err.println("索引創建失敗: " + createIndexResponse.getException().getMessage());
        } else {
            System.out.println("索引創建成功。");
        }
    }

    // 3. 加載集合
    public void loadCollection() {
        R<RpcStatus> loadResponse = milvusClient.loadCollection(
                LoadCollectionParam.newBuilder()
                        .withCollectionName("ticket_collection")
                        .build()
        );

        if (loadResponse.getStatus() != 0) {
            System.err.println("加載集合失敗: " + loadResponse.getException().getMessage());
        } else {
            System.out.println("集合加載成功。");
        }
    }

    // 4. 相似性搜索
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

        // 呼叫嵌入模型生成向量資料
        EmbeddingRequest request = new EmbeddingRequest(ticketInputs, null);
        EmbeddingResponse response = embeddingModel.call(request);

        // 將 float[] 轉換為 List<Float>
        float[] embeddingOutput = response.getResults().get(0).getOutput();
        List<Float> queryVector = adjustVectorDimensions(embeddingOutput, 128);

        // 構建搜索參數
        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName("ticket_collection")
                .withMetricType(MetricType.L2)
                .withTopK(topK)
                .withVectors(List.of(queryVector))
                .withVectorFieldName("vector")
                .withOutFields(List.of("userId", "trainId", "departure", "destination", "bookingTime")) // 指定需要返回的字段
                .build();

        // 執行搜索
        R<SearchResults> searchResponse = milvusClient.search(searchParam);
        if (searchResponse.getStatus() == 0 && searchResponse.getData() != null) {
            SearchResultsWrapper wrapper = new SearchResultsWrapper(searchResponse.getData().getResults());
            return wrapper.getIDScore(0);
        } else {
            System.err.println("搜索失敗: " + searchResponse.getException().getMessage());
            return null;
        }
    }

    // 工具方法: 將 float[] 轉換為 List<Float>
    private List<Float> floatArrayToList(float[] floatArray) {
        List<Float> list = new ArrayList<>(floatArray.length);
        for (float f : floatArray) {
            list.add(f);
        }
        return list;
    }

    private List<Float> adjustVectorDimensions(float[] embeddingOutput, int requiredDimension) {
        List<Float> vector = floatArrayToList(embeddingOutput);
        // 如果向量長度不足，則補 0
        while (vector.size() < requiredDimension) {
            vector.add(0.0f);  // 使用 0 來填充
        }
        // 確保最終向量長度正好為 requiredDimension
        return vector.subList(0, requiredDimension);
    }
}
