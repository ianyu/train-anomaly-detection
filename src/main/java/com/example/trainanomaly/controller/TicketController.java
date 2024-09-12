package com.example.trainanomaly.controller;

import com.example.trainanomaly.model.Ticket;
import com.example.trainanomaly.service.MilvusService;
import io.milvus.response.SearchResultsWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/tickets")
public class TicketController {

    @Autowired
    private MilvusService milvusService;

    @PostMapping("/insert")
    public ResponseEntity<String> insertTicket(@RequestBody Ticket ticket) {
        milvusService.insertTicket(ticket);
        return ResponseEntity.ok("Ticket inserted successfully.");
    }

    @PostMapping("/search")
    public ResponseEntity<List<SearchResultsWrapper.IDScore>> searchSimilarTickets(@RequestBody Ticket ticket, @RequestParam(defaultValue = "5") int topK) {
        List<SearchResultsWrapper.IDScore> results = milvusService.searchSimilarTickets(ticket, topK);
        return ResponseEntity.ok(results);
    }
}
