package com.example.trainanomaly.service;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.embedding.Embedding;
import org.springframework.util.Assert;
import java.util.List;
import java.util.ArrayList;

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

    @Override
    public float[] embed(Document document) {
        Assert.notNull(document, "Document must not be null");
        String content = document.getContent();
        return generateEmbedding(content);
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
