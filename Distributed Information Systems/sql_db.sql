-- phpMyAdmin SQL Dump
-- version 5.1.1deb5ubuntu1
-- https://www.phpmyadmin.net/
--
-- Host: devweb2023.cis.strath.ac.uk:3306
-- Generation Time: Apr 04, 2024 at 10:20 PM
-- Server version: 8.0.36-0ubuntu0.22.04.1
-- PHP Version: 8.1.2-1ubuntu2.14

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `dsb24101`
--

-- --------------------------------------------------------

--
-- Table structure for table `Department`
--

CREATE TABLE `Department` (
  `Department_ID` varchar(255) NOT NULL,
  `Name` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `Department`
--

INSERT INTO `Department` (`Department_ID`, `Name`) VALUES
('Dept01', 'Group18 Technical'),
('Dept02', 'Group 18 Management'),
('Dept03', 'Group19 Technical'),
('Dept04', 'Group19 Management'),
('Dept05', 'Group20 Technical');

-- --------------------------------------------------------

--
-- Table structure for table `Employee`
--

CREATE TABLE `Employee` (
  `Employee_ID` varchar(255) NOT NULL,
  `Name` varchar(255) DEFAULT NULL,
  `Role` varchar(255) DEFAULT NULL,
  `Job_Type` varchar(255) DEFAULT NULL,
  `Department_ID` varchar(255) DEFAULT NULL,
  `Office_ID` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `Employee`
--

INSERT INTO `Employee` (`Employee_ID`, `Name`, `Role`, `Job_Type`, `Department_ID`, `Office_ID`) VALUES
('Emp01', 'Mohan', 'App Developer', 'Software', 'Dept01', 'Office01'),
('Emp02', 'Harshan', 'App Developer', 'Software', 'Dept01', 'Office01'),
('Emp03', 'Prithvi', 'App Developer', 'Software', 'Dept01', 'Office01'),
('Emp04', 'Manoj', 'HR', 'Management', 'Dept02', 'Office01'),
('Emp05', 'Valerio', 'HR', 'Management', 'Dept02', 'Office01');

-- --------------------------------------------------------

--
-- Table structure for table `Office`
--

CREATE TABLE `Office` (
  `Office_ID` varchar(255) NOT NULL,
  `Location` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `Office`
--

INSERT INTO `Office` (`Office_ID`, `Location`) VALUES
('Office01', 'Glasgow'),
('Office02', 'Chennai'),
('Office03', 'Banglore'),
('Office04', 'London'),
('Office05', 'New York');

-- --------------------------------------------------------

--
-- Table structure for table `Project`
--

CREATE TABLE `Project` (
  `Project_ID` varchar(255) NOT NULL,
  `Name` varchar(255) DEFAULT NULL,
  `Department_ID` varchar(255) DEFAULT NULL,
  `Type` varchar(255) NOT NULL,
  `Category` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `Project`
--

INSERT INTO `Project` (`Project_ID`, `Name`, `Department_ID`, `Type`, `Category`) VALUES
('Proj01', 'DIS App', 'Dept01', 'Mobile App', 'Educational Software'),
('Proj02', 'ML App', 'Dept01', 'Mobile App', 'Educational Software'),
('Proj03', 'BA App', 'Dept01', 'Mobile App', 'Educational Software'),
('Proj04', 'DIS Website', 'Dept01', 'Web App', 'Educational Software'),
('Proj05', 'ML Website', 'Dept01', 'Web App', 'Educational Software');

-- --------------------------------------------------------

--
-- Table structure for table `Project_Record`
--

CREATE TABLE `Project_Record` (
  `Record_ID` varchar(255) DEFAULT NULL,
  `Month_ID` int DEFAULT NULL,
  `Completion_` varchar(255) DEFAULT NULL,
  `Project_ID` varchar(255) DEFAULT NULL,
  `Employee_ID` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `Project_Record`
--

INSERT INTO `Project_Record` (`Record_ID`, `Month_ID`, `Completion_`, `Project_ID`, `Employee_ID`) VALUES
('Record01', 4, 'Pending', 'Proj01', 'Emp01'),
('Record02', 4, 'Pending', 'Proj01', 'Emp02'),
('Record03', 4, 'Pending', 'Proj01', 'Emp03'),
('Record42', 5, 'Completed', 'Proj01', 'Emp01'),
('Record05', 5, 'Completed', 'Proj01', 'Emp02');

-- --------------------------------------------------------

--
-- Table structure for table `Task`
--

CREATE TABLE `Task` (
  `Task_Code` varchar(255) NOT NULL,
  `Name` varchar(255) DEFAULT NULL,
  `Weight` int DEFAULT NULL,
  `Project_ID` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `Task`
--

INSERT INTO `Task` (`Task_Code`, `Name`, `Weight`, `Project_ID`) VALUES
('Task01', 'Front-end', 30, 'Proj01'),
('Task02', 'Back-End', 30, 'Proj01'),
('Task03', 'API', 20, 'Proj01'),
('Task04', 'User Testing', 10, 'Proj01'),
('Task05', 'Automated Testing', 10, 'Proj01');

-- --------------------------------------------------------

--
-- Table structure for table `Team`
--

CREATE TABLE `Team` (
  `Team_ID` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `Team`
--

INSERT INTO `Team` (`Team_ID`) VALUES
('Team01'),
('Team02'),
('Team03'),
('Team04'),
('Team05');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `Department`
--
ALTER TABLE `Department`
  ADD PRIMARY KEY (`Department_ID`);

--
-- Indexes for table `Employee`
--
ALTER TABLE `Employee`
  ADD PRIMARY KEY (`Employee_ID`),
  ADD KEY `Department_ID` (`Department_ID`),
  ADD KEY `Office_ID` (`Office_ID`);

--
-- Indexes for table `Office`
--
ALTER TABLE `Office`
  ADD PRIMARY KEY (`Office_ID`);

--
-- Indexes for table `Project`
--
ALTER TABLE `Project`
  ADD PRIMARY KEY (`Project_ID`),
  ADD KEY `Department_ID` (`Department_ID`);

--
-- Indexes for table `Project_Record`
--
ALTER TABLE `Project_Record`
  ADD KEY `Project_ID` (`Project_ID`),
  ADD KEY `Employee_ID` (`Employee_ID`);

--
-- Indexes for table `Task`
--
ALTER TABLE `Task`
  ADD PRIMARY KEY (`Task_Code`),
  ADD KEY `Project_ID` (`Project_ID`);

--
-- Constraints for dumped tables
--

--
-- Constraints for table `Employee`
--
ALTER TABLE `Employee`
  ADD CONSTRAINT `Employee_ibfk_1` FOREIGN KEY (`Department_ID`) REFERENCES `Department` (`Department_ID`),
  ADD CONSTRAINT `Employee_ibfk_2` FOREIGN KEY (`Office_ID`) REFERENCES `Office` (`Office_ID`);

--
-- Constraints for table `Project`
--
ALTER TABLE `Project`
  ADD CONSTRAINT `Project_ibfk_1` FOREIGN KEY (`Department_ID`) REFERENCES `Department` (`Department_ID`);

--
-- Constraints for table `Project_Record`
--
ALTER TABLE `Project_Record`
  ADD CONSTRAINT `Project_Record_ibfk_1` FOREIGN KEY (`Project_ID`) REFERENCES `Project` (`Project_ID`),
  ADD CONSTRAINT `Project_Record_ibfk_2` FOREIGN KEY (`Employee_ID`) REFERENCES `Employee` (`Employee_ID`);

--
-- Constraints for table `Task`
--
ALTER TABLE `Task`
  ADD CONSTRAINT `Task_ibfk_1` FOREIGN KEY (`Project_ID`) REFERENCES `Project` (`Project_ID`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
