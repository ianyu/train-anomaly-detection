package com.example.trainanomaly.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Ticket {
    private String userId;
    private String trainId;
    private String departure;
    private String destination;
    private String bookingTime;
    // 其他相關欄位
}
