package edu.zjunb.llm.yeya.controller;

import edu.zjunb.llm.yeya.service.FileProcessingService;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.beans.factory.annotation.Autowired;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@RestController
@RequestMapping("/api")
public class FileUploadController {

    private static final String UPLOAD_DIR = "uploads/";

    @Autowired
    private FileProcessingService fileProcessingService;

    @PostMapping("/upload")
    public ResponseEntity<Map<String, Object>> uploadFiles(@RequestParam("files") MultipartFile[] files) {
        Map<String, Object> response = new HashMap<>();
        List<Map<String, Object>> fileDetails = new ArrayList<>();

        if (files.length == 0) {
            response.put("message", "No files to upload");
            return new ResponseEntity<>(response, HttpStatus.BAD_REQUEST);
        }

        try {
            // Ensure the upload directory exists
            Path uploadPath = Paths.get(UPLOAD_DIR);
            if (!Files.exists(uploadPath)) {
                Files.createDirectories(uploadPath);
            }

            for (MultipartFile file : files) {
                if (file.isEmpty()) {
                    continue;
                }
                // Save each file locally
                LocalDateTime now = LocalDateTime.now();
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyMMddHHmmss");
                String formattedNow = now.format(formatter);
                byte[] bytes = file.getBytes();
                String fileName=formattedNow+"_"+file.getOriginalFilename();
                Path path = Paths.get(UPLOAD_DIR + fileName);
                Files.write(path, bytes);
                // 异步处理文件
                fileProcessingService.processFileAsync(path.toAbsolutePath().toString());

                // Collect file details
                Map<String, Object> fileDetail = new HashMap<>();
                fileDetail.put("fileName", fileName);
                fileDetail.put("fileSize", file.getSize());
                // fileDetail.put("fileUri", path.toUri().toString());
                fileDetails.add(fileDetail);
            }

            response.put("message", "Files uploaded successfully");
            response.put("fileCount", files.length);
            response.put("files", fileDetails);
            return new ResponseEntity<>(response, HttpStatus.OK);
        } catch (IOException e) {
            e.printStackTrace();
            response.put("message", "Failed to upload files");
            return new ResponseEntity<>(response, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
}