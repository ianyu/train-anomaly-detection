package com.example.trainanomaly.config;

import com.example.trainanomaly.service.CustomEmbeddingModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.ai.embedding.EmbeddingModel;

@Configuration
public class EmbeddingModelConfig {

    // 定義一個自訂的 EmbeddingModel Bean
    @Bean
    public EmbeddingModel embeddingModel() {
        return new CustomEmbeddingModel();
    }
}
