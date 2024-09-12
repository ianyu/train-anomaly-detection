package com.example.trainanomaly.service;

import io.milvus.client.MilvusServiceClient;
import io.milvus.param.collection.*;
import io.milvus.grpc.DataType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import jakarta.annotation.PostConstruct;

@Service
public class MilvusCollectionService {

    @Autowired
    private MilvusServiceClient milvusClient;

    private static final String COLLECTION_NAME = "ticket_collection";

    @PostConstruct
    public void init() {
        createCollectionIfNotExists();
    }

    public void createCollectionIfNotExists() {
        HasCollectionParam hasCollectionParam = HasCollectionParam.newBuilder()
                .withCollectionName(COLLECTION_NAME)
                .build();

        boolean hasCollection = milvusClient.hasCollection(hasCollectionParam).getData();

        if (!hasCollection) {
            FieldType idField = FieldType.newBuilder()
                    .withName("id")
                    .withDataType(DataType.Int64)
                    .withPrimaryKey(true)
                    .withAutoID(true)
                    .build();

            FieldType vectorField = FieldType.newBuilder()
                    .withName("vector")
                    .withDataType(DataType.FloatVector)
                    .withDimension(128)
                    .build();
// 定義其他字段，例如 userId, trainId, departure, destination, bookingTime
            FieldType userIdField = FieldType.newBuilder()
                    .withName("userId")
                    .withDataType(DataType.VarChar)
                    .withMaxLength(50)  // 指定最大長度
                    .build();

            FieldType trainIdField = FieldType.newBuilder()
                    .withName("trainId")
                    .withDataType(DataType.VarChar)
                    .withMaxLength(50)  // 指定最大長度
                    .build();

            FieldType departureField = FieldType.newBuilder()
                    .withName("departure")
                    .withDataType(DataType.VarChar)
                    .withMaxLength(50)  // 指定最大長度
                    .build();

            FieldType destinationField = FieldType.newBuilder()
                    .withName("destination")
                    .withDataType(DataType.VarChar)
                    .withMaxLength(50)  // 指定最大長度
                    .build();

            FieldType bookingTimeField = FieldType.newBuilder()
                    .withName("bookingTime")
                    .withDataType(DataType.VarChar)
                    .withMaxLength(50)  // 指定最大長度
                    .build();

            CreateCollectionParam createCollectionParam = CreateCollectionParam.newBuilder()
                    .withCollectionName(COLLECTION_NAME)
                    .withShardsNum(2)
                    .addFieldType(idField)
                    .addFieldType(vectorField)
                    .addFieldType(userIdField)       // 添加其他字段
                    .addFieldType(trainIdField)
                    .addFieldType(departureField)
                    .addFieldType(destinationField)
                    .addFieldType(bookingTimeField)
                    .build();

            milvusClient.createCollection(createCollectionParam);
            System.out.println("Collection created: " + COLLECTION_NAME);
        } else {
            System.out.println("Collection already exists: " + COLLECTION_NAME);
        }
    }
}
