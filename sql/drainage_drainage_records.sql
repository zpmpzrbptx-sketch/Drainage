-- MySQL dump 10.13  Distrib 8.0.46, for macos15 (arm64)
--
-- Host: localhost    Database: drainage
-- ------------------------------------------------------
-- Server version	9.7.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
SET @MYSQLDUMP_TEMP_LOG_BIN = @@SESSION.SQL_LOG_BIN;
SET @@SESSION.SQL_LOG_BIN= 0;

--
-- GTID state at the beginning of the backup 
--

SET @@GLOBAL.GTID_PURGED=/*!80000 '+'*/ '7a5da010-5a57-11f1-b2d4-178553125fa7:1-933';

--
-- Table structure for table `drainage_records`
--

DROP TABLE IF EXISTS `drainage_records`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `drainage_records` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `scenario` varchar(64) NOT NULL DEFAULT 'default',
  `record_time` datetime NOT NULL,
  `rain` double NOT NULL DEFAULT '0',
  `water_node1` double NOT NULL DEFAULT '0',
  `water_node2` double NOT NULL DEFAULT '0',
  `water_node3` double NOT NULL DEFAULT '0',
  `flow_node1` double NOT NULL DEFAULT '0',
  `flow_node2` double NOT NULL DEFAULT '0',
  `flow_node3` double NOT NULL DEFAULT '0',
  `storage_node1` double NOT NULL DEFAULT '0',
  `storage_node2` double NOT NULL DEFAULT '0',
  `storage_node3` double NOT NULL DEFAULT '0',
  `energy` double NOT NULL DEFAULT '0',
  `overflow` double NOT NULL DEFAULT '0',
  `reward` double NOT NULL DEFAULT '0',
  `risk_level` enum('low','medium','high') NOT NULL DEFAULT 'low',
  `mode` varchar(32) NOT NULL DEFAULT 'rule',
  `remark` varchar(255) DEFAULT NULL,
  `created_by` bigint DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_record_time` (`record_time`),
  KEY `idx_scenario` (`scenario`),
  KEY `idx_risk_level` (`risk_level`),
  KEY `idx_mode` (`mode`),
  KEY `fk_drainage_created_by` (`created_by`),
  CONSTRAINT `fk_drainage_created_by` FOREIGN KEY (`created_by`) REFERENCES `admin_users` (`id`) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `drainage_records`
--

LOCK TABLES `drainage_records` WRITE;
/*!40000 ALTER TABLE `drainage_records` DISABLE KEYS */;
/*!40000 ALTER TABLE `drainage_records` ENABLE KEYS */;
UNLOCK TABLES;
SET @@SESSION.SQL_LOG_BIN = @MYSQLDUMP_TEMP_LOG_BIN;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-06-11 18:56:19
