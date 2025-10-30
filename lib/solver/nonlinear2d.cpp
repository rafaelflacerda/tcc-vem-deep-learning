#include "solver/nonlinear2d.hpp"

double solver::nonlinear2d::calculateTaylor5(double nu) {
    double term = 0.25 + nu;
    return -444.444444444444 + 
           1777.7777777777776 * term - 
           790.1234567901234 * std::pow(term, 2) + 
           3160.4938271604938 * std::pow(term, 3) - 
           1404.6639231824408 * std::pow(term, 4) + 
           5618.655692729764 * std::pow(term, 5);
}


void solver::nonlinear2d::localKcTriangle(const Eigen::VectorXd& L, const double* A, const double* Mu, const double* La, const Eigen::VectorXd& u, const Eigen::MatrixXd& n, Eigen::MatrixXd& Kc){

    Eigen::VectorXd v(284); // Adjust the size as needed

    v(49) = L(0) / 2.0;
    v(51) = L(1) / 2.0;
    v(50) = L(2) / 2.0;
    v(4) = *A;
    v(168) = v(51) / v(4);
    v(167) = v(50) / v(4);
    v(166) = v(49) / v(4);
    v(5) = *Mu;
    v(164) = v(4) * v(5);
    v(6) = *La;
    v(7) = u(0);
    v(8) = u(1);
    v(9) = u(2);
    v(20) = v(49) * (v(7) + v(9));
    v(10) = u(3);
    v(25) = v(49) * (v(10) + v(8));
    v(11) = u(4);
    v(22) = v(50) * (v(11) + v(7));
    v(21) = v(51) * (v(11) + v(9));
    v(12) = u(5);
    v(27) = v(50) * (v(12) + v(8));
    v(26) = (v(10) + v(12)) * v(51);
    v(13) = n(0, 0);
    v(169) = v(13) / v(4);
    v(53) = v(13) * v(49);
    v(14) = n(0, 1);
    v(170) = v(14) / v(4);
    v(59) = v(14) * v(49);
    v(15) = n(1, 0);
    v(173) = v(15) / v(4);
    v(56) = v(15) * v(51);
    v(54) = (v(53) + v(56)) / v(4);
    v(95) = v(164) * v(54);
    v(16) = n(1, 1);
    v(174) = v(16) / v(4);
    v(62) = v(16) * v(51);
    v(60) = (v(59) + v(62)) / v(4);
    v(105) = v(164) * v(60);
    v(17) = n(2, 0);
    v(171) = v(17) / v(4);
    v(55) = v(17) * v(50);
    v(57) = (v(55) + v(56)) / v(4);
    v(99) = v(164) * v(57);
    v(52) = (v(53) + v(55)) / v(4);
    v(91) = v(164) * v(52);
    v(18) = n(2, 1);
    v(172) = v(18) / v(4);
    v(61) = v(18) * v(50);
    v(63) = (v(61) + v(62)) / v(4);
    v(109) = v(164) * v(63);
    v(58) = (v(59) + v(61)) / v(4);
    v(101) = v(164) * v(58);
    v(23) = (v(14) * v(20) + v(16) * v(21) + v(18) * v(22)) / v(4);
    v(24) = (v(13) * v(25) + v(15) * v(26) + v(17) * v(27)) / v(4);
    v(29) = 1.0 + (v(13) * v(20) + v(15) * v(21) + v(17) * v(22)) / v(4);
    v(30) = 1.0 + (v(14) * v(25) + v(16) * v(26) + v(18) * v(27)) / v(4);
    v(31) = -(v(23) * v(24)) + v(29) * v(30);
    v(165) = v(4) * (v(5) / (v(31) * v(31)) + v(6));
    v(76) = v(165) * (-(v(23) * v(57)) + v(29) * v(63));
    v(100) = -(v(23) * v(76)) + v(99);
    v(82) = v(109) + v(29) * v(76);
    v(75) = v(165) * (v(30) * v(57) - v(24) * v(63));
    v(110) = v(109) - v(24) * v(75);
    v(87) = v(30) * v(75) + v(99);
    v(74) = v(165) * (-(v(23) * v(54)) + v(29) * v(60));
    v(96) = -(v(23) * v(74)) + v(95);
    v(80) = v(105) + v(29) * v(74);
    v(73) = v(165) * (v(30) * v(54) - v(24) * v(60));
    v(106) = v(105) - v(24) * v(73);
    v(85) = v(30) * v(73) + v(95);
    v(72) = v(165) * (-(v(23) * v(52)) + v(29) * v(58));
    v(92) = -(v(23) * v(72)) + v(91);
    v(78) = v(101) + v(29) * v(72);
    v(70) = v(165) * (v(30) * v(52) - v(24) * v(58));
    v(102) = v(101) - v(24) * v(70);
    v(83) = v(30) * v(70) + v(91);
    v(38) = v(4) * (-(v(5) / v(31)) + (-1.0 + v(31)) * v(6));
    v(111) = v(38) * v(57);
    v(112) = -v(111) - v(24) * v(76);
    v(107) = v(38) * v(54);
    v(108) = -v(107) - v(24) * v(74);
    v(104) = -v(38) * v(52) - v(24) * v(72);
    v(97) = v(38) * v(63);
    v(98) = -(v(23) * v(75)) - v(97);
    v(93) = v(38) * v(60);
    v(94) = -(v(23) * v(73)) - v(93);
    v(88) = v(30) * v(76) + v(97);
    v(86) = v(30) * v(74) + v(93);
    v(84) = v(38) * v(58) + v(30) * v(72);
    v(81) = v(111) + v(29) * v(75);
    v(79) = v(107) + v(29) * v(73);
    v(149) = v(166) * (v(106) * v(14) + v(13) * v(85));
    v(150) = v(166) * (v(108) * v(14) + v(13) * v(86));
    v(151) = v(167) * (v(110) * v(18) + v(17) * v(87));
    v(152) = v(166) * (v(110) * v(14) + v(13) * v(87));
    v(153) = v(167) * (v(112) * v(18) + v(17) * v(88));
    v(154) = v(166) * (v(112) * v(14) + v(13) * v(88));
    v(155) = v(166) * (v(14) * v(80) + v(13) * v(96));
    v(156) = v(166) * (v(14) * v(81) + v(13) * v(98));
    v(157) = v(167) * (v(100) * v(17) + v(18) * v(82));
    v(158) = v(166) * (v(100) * v(13) + v(14) * v(82));
    v(159) = v(168) * (v(110) * v(16) + v(15) * v(87));
    v(160) = v(168) * (v(112) * v(16) + v(15) * v(88));
    v(161) = v(168) * (v(100) * v(15) + v(16) * v(82));
    Kc(0, 0) = v(49) * (v(102) * v(170) + v(169) * v(83)) + v(50) * (v(102) * v(172) + v(171) * v(83));
    Kc(0, 1) = v(49) * (v(104) * v(170) + v(169) * v(84)) + v(50) * (v(104) * v(172) + v(171) * v(84));
    Kc(0, 2) = v(149) + v(50) * (v(106) * v(172) + v(171) * v(85));
    Kc(0, 3) = v(150) + v(50) * (v(108) * v(172) + v(171) * v(86));
    Kc(0, 4) = v(151) + v(152);
    Kc(0, 5) = v(153) + v(154);
    Kc(1, 1) = v(49) * (v(170) * v(78) + v(169) * v(92)) + v(50) * (v(172) * v(78) + v(171) * v(92));
    Kc(1, 2) = v(49) * (v(170) * v(79) + v(169) * v(94)) + v(50) * (v(172) * v(79) + v(171) * v(94));
    Kc(1, 3) = v(155) + v(50) * (v(172) * v(80) + v(171) * v(96));
    Kc(1, 4) = v(156) + v(50) * (v(172) * v(81) + v(171) * v(98));
    Kc(1, 5) = v(157) + v(158);
    Kc(2, 2) = v(149) + v(51) * (v(106) * v(174) + v(173) * v(85));
    Kc(2, 3) = v(150) + v(51) * (v(108) * v(174) + v(173) * v(86));
    Kc(2, 4) = v(152) + v(159);
    Kc(2, 5) = v(154) + v(160);
    Kc(3, 3) = v(155) + v(51) * (v(174) * v(80) + v(173) * v(96));
    Kc(3, 4) = v(156) + v(51) * (v(174) * v(81) + v(173) * v(98));
    Kc(3, 5) = v(158) + v(161);
    Kc(4, 4) = v(151) + v(159);
    Kc(4, 5) = v(153) + v(160);
    Kc(5, 5) = v(157) + v(161);
}

void solver::nonlinear2d::localKcQuadrilateral(const Eigen::VectorXd& L, const double* A, const double* Mu, const double* La, const Eigen::VectorXd& u, const Eigen::MatrixXd& n, Eigen::MatrixXd& Kc){

    Eigen::VectorXd v(250); // Adjust the size as needed

    v(58) = L(0) / 2.0;
    v(59) = L(1) / 2.0;
    v(61) = L(2) / 2.0;
    v(60) = L(3) / 2.0;
    v(5) = (*A);
    v(238) = v(61) / v(5);
    v(237) = v(59) / v(5);
    v(236) = v(60) / v(5);
    v(235) = v(58) / v(5);
    v(6) = (*Mu);
    v(233) = v(5) * v(6);
    v(7) = (*La);
    v(8) = u(0);
    v(9) = u(1);
    v(10) = u(2);
    v(25) = v(58) * (v(10) + v(8));
    v(11) = u(3);
    v(31) = v(58) * (v(11) + v(9));
    v(12) = u(4);
    v(26) = (v(10) + v(12)) * v(59);
    v(13) = u(5);
    v(32) = (v(11) + v(13)) * v(59);
    v(14) = u(6);
    v(28) = v(60) * (v(14) + v(8));
    v(27) = (v(12) + v(14)) * v(61);
    v(15) = u(7);
    v(34) = v(60) * (v(15) + v(9));
    v(33) = (v(13) + v(15)) * v(61);
    v(16) = n(0, 0);
    v(239) = v(16) / v(5);
    v(63) = v(16) * v(58);
    v(17) = n(0, 1);
    v(240) = v(17) / v(5);
    v(71) = v(17) * v(58);
    v(18) = n(1, 0);
    v(243) = v(18) / v(5);
    v(65) = v(18) * v(59);
    v(64) = (v(63) + v(65)) / v(5);
    v(117) = v(233) * v(64);
    v(19) = n(1, 1);
    v(244) = v(19) / v(5);
    v(73) = v(19) * v(59);
    v(72) = (v(71) + v(73)) / v(5);
    v(131) = v(233) * v(72);
    v(20) = n(2, 0);
    v(245) = v(20) / v(5);
    v(68) = v(20) * v(61);
    v(66) = (v(65) + v(68)) / v(5);
    v(121) = v(233) * v(66);
    v(21) = n(2, 1);
    v(246) = v(21) / v(5);
    v(76) = v(21) * v(61);
    v(74) = (v(73) + v(76)) / v(5);
    v(135) = v(233) * v(74);
    v(22) = n(3, 0);
    v(241) = v(22) / v(5);
    v(67) = v(22) * v(60);
    v(69) = (v(67) + v(68)) / v(5);
    v(125) = v(233) * v(69);
    v(62) = (v(63) + v(67)) / v(5);
    v(113) = v(233) * v(62);
    v(23) = n(3, 1);
    v(242) = v(23) / v(5);
    v(75) = v(23) * v(60);
    v(77) = (v(75) + v(76)) / v(5);
    v(139) = v(233) * v(77);
    v(70) = (v(71) + v(75)) / v(5);
    v(127) = v(233) * v(70);
    v(29) = (v(17) * v(25) + v(19) * v(26) + v(21) * v(27) + v(23) * v(28)) / v(5);
    v(30) = (v(16) * v(31) + v(18) * v(32) + v(20) * v(33) + v(22) * v(34)) / v(5);
    v(36) = 1.0 + (v(16) * v(25) + v(18) * v(26) + v(20) * v(27) + v(22) * v(28)) / v(5);
    v(37) = 1.0 + (v(17) * v(31) + v(19) * v(32) + v(21) * v(33) + v(23) * v(34)) / v(5);
    v(38) = -(v(29) * v(30)) + v(36) * v(37);
    v(234) = v(5) * (v(6) / (v(38) * v(38)) + v(7));
    v(94) = v(234) * (-(v(29) * v(69)) + v(36) * v(77));
    v(126) = v(125) - v(29) * v(94);
    v(102) = v(139) + v(36) * v(94);
    v(93) = v(234) * (v(37) * v(69) - v(30) * v(77));
    v(140) = v(139) - v(30) * v(93);
    v(109) = v(125) + v(37) * v(93);
    v(92) = v(234) * (-(v(29) * v(66)) + v(36) * v(74));
    v(122) = v(121) - v(29) * v(92);
    v(100) = v(135) + v(36) * v(92);
    v(91) = v(234) * (v(37) * v(66) - v(30) * v(74));
    v(136) = v(135) - v(30) * v(91);
    v(107) = v(121) + v(37) * v(91);
    v(90) = v(234) * (-(v(29) * v(64)) + v(36) * v(72));
    v(118) = v(117) - v(29) * v(90);
    v(98) = v(131) + v(36) * v(90);
    v(89) = v(234) * (v(37) * v(64) - v(30) * v(72));
    v(132) = v(131) - v(30) * v(89);
    v(105) = v(117) + v(37) * v(89);
    v(88) = v(234) * (-(v(29) * v(62)) + v(36) * v(70));
    v(114) = v(113) - v(29) * v(88);
    v(96) = v(127) + v(36) * v(88);
    v(86) = v(234) * (v(37) * v(62) - v(30) * v(70));
    v(128) = v(127) - v(30) * v(86);
    v(103) = v(113) + v(37) * v(86);
    v(45) = v(5) * (-(v(6) / v(38)) + (-1.0 + v(38)) * v(7));
    v(141) = v(45) * v(69);
    v(142) = -v(141) - v(30) * v(94);
    v(137) = v(45) * v(66);
    v(138) = -v(137) - v(30) * v(92);
    v(133) = v(45) * v(64);
    v(134) = -v(133) - v(30) * v(90);
    v(130) = -(v(45) * v(62)) - v(30) * v(88);
    v(123) = v(45) * v(77);
    v(124) = -v(123) - v(29) * v(93);
    v(119) = v(45) * v(74);
    v(120) = -v(119) - v(29) * v(91);
    v(115) = v(45) * v(72);
    v(116) = -v(115) - v(29) * v(89);
    v(110) = v(123) + v(37) * v(94);
    v(108) = v(119) + v(37) * v(92);
    v(106) = v(115) + v(37) * v(90);
    v(104) = v(45) * v(70) + v(37) * v(88);
    v(101) = v(141) + v(36) * v(93);
    v(99) = v(137) + v(36) * v(91);
    v(97) = v(133) + v(36) * v(89);
    v(207) = (v(105) * v(16) + v(132) * v(17)) * v(235);
    v(208) = (v(106) * v(16) + v(134) * v(17)) * v(235);
    v(209) = (v(107) * v(16) + v(136) * v(17)) * v(235);
    v(210) = (v(108) * v(16) + v(138) * v(17)) * v(235);
    v(211) = (v(109) * v(22) + v(140) * v(23)) * v(236);
    v(212) = (v(109) * v(16) + v(140) * v(17)) * v(235);
    v(213) = (v(110) * v(22) + v(142) * v(23)) * v(236);
    v(214) = (v(110) * v(16) + v(142) * v(17)) * v(235);
    v(215) = v(235) * (v(118) * v(16) + v(17) * v(98));
    v(216) = v(235) * (v(120) * v(16) + v(17) * v(99));
    v(217) = (v(122) * v(16) + v(100) * v(17)) * v(235);
    v(218) = (v(124) * v(16) + v(101) * v(17)) * v(235);
    v(219) = (v(126) * v(22) + v(102) * v(23)) * v(236);
    v(220) = (v(126) * v(16) + v(102) * v(17)) * v(235);
    v(221) = (v(107) * v(18) + v(136) * v(19)) * v(237);
    v(222) = (v(108) * v(18) + v(138) * v(19)) * v(237);
    v(223) = (v(109) * v(18) + v(140) * v(19)) * v(237);
    v(224) = (v(110) * v(18) + v(142) * v(19)) * v(237);
    v(225) = (v(122) * v(18) + v(100) * v(19)) * v(237);
    v(226) = (v(124) * v(18) + v(101) * v(19)) * v(237);
    v(227) = (v(126) * v(18) + v(102) * v(19)) * v(237);
    v(228) = (v(109) * v(20) + v(140) * v(21)) * v(238);
    v(229) = (v(110) * v(20) + v(142) * v(21)) * v(238);
    v(230) = (v(126) * v(20) + v(102) * v(21)) * v(238);
    Kc(0, 0) = (v(103) * v(239) + v(128) * v(240)) * v(58) + (v(103) * v(241) + v(128) * v(242)) * v(60);
    Kc(0, 1) = (v(104) * v(239) + v(130) * v(240)) * v(58) + (v(104) * v(241) + v(130) * v(242)) * v(60);
    Kc(0, 2) = v(207) + (v(105) * v(241) + v(132) * v(242)) * v(60);
    Kc(0, 3) = v(208) + (v(106) * v(241) + v(134) * v(242)) * v(60);
    Kc(0, 4) = v(209) + (v(107) * v(241) + v(136) * v(242)) * v(60);
    Kc(0, 5) = v(210) + (v(108) * v(241) + v(138) * v(242)) * v(60);
    Kc(0, 6) = v(211) + v(212);
    Kc(0, 7) = v(213) + v(214);
    Kc(1, 1) = v(58) * (v(114) * v(239) + v(240) * v(96)) + v(60) * (v(114) * v(241) + v(242) * v(96));
    Kc(1, 2) = v(58) * (v(116) * v(239) + v(240) * v(97)) + v(60) * (v(116) * v(241) + v(242) * v(97));
    Kc(1, 3) = v(215) + v(60) * (v(118) * v(241) + v(242) * v(98));
    Kc(1, 4) = v(216) + v(60) * (v(120) * v(241) + v(242) * v(99));
    Kc(1, 5) = v(217) + (v(122) * v(241) + v(100) * v(242)) * v(60);
    Kc(1, 6) = v(218) + (v(124) * v(241) + v(101) * v(242)) * v(60);
    Kc(1, 7) = v(219) + v(220);
    Kc(2, 2) = v(207) + (v(105) * v(243) + v(132) * v(244)) * v(59);
    Kc(2, 3) = v(208) + (v(106) * v(243) + v(134) * v(244)) * v(59);
    Kc(2, 4) = v(209) + v(221);
    Kc(2, 5) = v(210) + v(222);
    Kc(2, 6) = v(212) + v(223);
    Kc(2, 7) = v(214) + v(224);
    Kc(3, 3) = v(215) + v(59) * (v(118) * v(243) + v(244) * v(98));
    Kc(3, 4) = v(216) + v(59) * (v(120) * v(243) + v(244) * v(99));
    Kc(3, 5) = v(217) + v(225);
    Kc(3, 6) = v(218) + v(226);
    Kc(3, 7) = v(220) + v(227);
    Kc(4, 4) = v(221) + (v(107) * v(245) + v(136) * v(246)) * v(61);
    Kc(4, 5) = v(222) + (v(108) * v(245) + v(138) * v(246)) * v(61);
    Kc(4, 6) = v(223) + v(228);
    Kc(4, 7) = v(224) + v(229);
    Kc(5, 5) = v(225) + (v(122) * v(245) + v(100) * v(246)) * v(61);
    Kc(5, 6) = v(226) + (v(124) * v(245) + v(101) * v(246)) * v(61);
    Kc(5, 7) = v(227) + v(230);
    Kc(6, 6) = v(211) + v(228);
    Kc(6, 7) = v(213) + v(229);
    Kc(7, 7) = v(219) + v(230);

};

void solver::nonlinear2d::localRcTriangle(const Eigen::VectorXd& L, double A, double Mu, double La, const Eigen::VectorXd& u, const Eigen::MatrixXd& n, Eigen::VectorXd& Rc){
    Eigen::VectorXd v(172);

    v(56) = L(0) / 2.0;
    v(58) = L(1) / 2.0;
    v(57) = L(2) / 2.0;
    v(4) = A;
    v(62) = v(58) / v(4);
    v(61) = v(56) / v(4);
    v(60) = v(57) / v(4);
    v(5) = Mu;
    v(59) = v(4) * v(5);

    v(20) = v(56) * (u(0) + u(2));
    v(25) = v(56) * (u(3) + u(1));
    v(22) = v(57) * (u(4) + u(0));
    v(21) = v(58) * (u(4) + u(2));
    v(27) = v(57) * (u(5) + u(1));
    v(26) = v(58) * (u(3) + u(5));

    v(23) = (n(0, 1) * v(20) + n(1, 1) * v(21) + n(2, 1) * v(22)) / v(4);
    v(24) = (n(0, 0) * v(25) + n(1, 0) * v(26) + n(2, 0) * v(27)) / v(4);
    v(29) = 1.0 + (n(0, 0) * v(20) + n(1, 0) * v(21) + n(2, 0) * v(22)) / v(4);
    v(30) = 1.0 + (n(0, 1) * v(25) + n(1, 1) * v(26) + n(2, 1) * v(27)) / v(4);

    v(31) = -(v(23) * v(24)) + v(29) * v(30);
    v(38) = v(4) * (La * (-1.0 + v(31)) - v(5) / v(31));
    v(39) = v(29) * v(38) + v(30) * v(59);
    v(40) = v(30) * v(38) + v(29) * v(59);
    v(41) = -(v(23) * v(38)) + v(24) * v(59);
    v(42) = -(v(24) * v(38)) + v(23) * v(59);

    Rc(0) = v(60) * (n(2, 0) * v(40) + n(2, 1) * v(42)) + v(61) * (n(0, 0) * v(40) + n(0, 1) * v(42));
    Rc(1) = v(60) * (n(2, 1) * v(39) + n(2, 0) * v(41)) + v(61) * (n(0, 1) * v(39) + n(0, 0) * v(41));
    Rc(2) = v(61) * (n(0, 0) * v(40) + n(0, 1) * v(42)) + v(62) * (n(1, 0) * v(40) + n(1, 1) * v(42));
    Rc(3) = v(61) * (n(0, 1) * v(39) + n(0, 0) * v(41)) + v(62) * (n(1, 1) * v(39) + n(1, 0) * v(41));
    Rc(4) = v(60) * (n(2, 0) * v(40) + n(2, 1) * v(42)) + v(62) * (n(1, 0) * v(40) + n(1, 1) * v(42));
    Rc(5) = v(60) * (n(2, 1) * v(39) + n(2, 0) * v(41)) + v(62) * (n(1, 1) * v(39) + n(1, 0) * v(41));

    // std::cout << "Rc(0) = " << Rc(0) << std::endl;
    // std::cout << "v(60) = " << v(39) << std::endl;
    // std::cout << "A = " << A << std::endl;
}

Eigen::VectorXd solver::nonlinear2d::buildLocalLoadVector(double L, double q, double p, const Eigen::VectorXd& u){

    Eigen::VectorXd Rt = Eigen::VectorXd::Zero(4);

    double halfL = L / 2.0;
    double qHalfL = q * halfL;
    double pHalfL = p * halfL;

    Rt(0) = qHalfL;
    Rt(1) = pHalfL;
    Rt(2) = qHalfL;
    Rt(3) = pHalfL;

    // std::cout << "Local Load Vector" << std::endl;
    // std::cout << Rt << "\n" << std::endl;

    return Rt;
};

Eigen::VectorXd solver::nonlinear2d::buildLocalRc(const Eigen::MatrixXd coords, Eigen::VectorXd u){
    int numVertex = elements.cols();

    // initalize consistency and stability residual vectors
    Eigen::VectorXd Rc = Eigen::VectorXd::Zero(2*numVertex);
    Eigen::VectorXd Rs = Eigen::VectorXd::Zero(2*numVertex);

    // geometric information
    double area;
    Eigen::VectorXd Lengths = Eigen::VectorXd::Zero(numVertex);
    Eigen::MatrixXd normal = Eigen::MatrixXd::Zero(numVertex, 2);

    // compute geometric information
    utils::operations op;
    area = op.calcArea(coords);

    for(int i=0; i<numVertex; i++){
        Eigen::MatrixXd startNode = coords.row(i);
        Eigen::MatrixXd endNode = coords.row((i+1)%numVertex);
        // std::cout << "endNode :: " << (i+1)%numVertex << std::endl;

        Eigen::MatrixXd edge = op.buildEdge(startNode, endNode);
        Lengths(i) = op.calcLength(edge);
        // std::cout << "Lengths :: " << Lengths(i) << std::endl;
        normal.row(i) = op.computerNormalVector(edge);
        // std::cout << "normal :: " << normal.row(i) << std::endl;
    }

    switch(numVertex) {
        case 3:
            localRcTriangle(Lengths, area, Mu, La, u, normal, Rc);
            // std::cout << "Local Residual Stiffness Vector" << std::endl;
            // std::cout << Rc << "\n" << std::endl;
            break;
        default:
            std::cout << "Only triangular elements are supported at the moment." << std::endl;
            break;
    }

    return Rc + Rs;
}

Eigen::MatrixXd solver::nonlinear2d::buildLocalK(Eigen::MatrixXd coords, Eigen::VectorXd u){
    int numVertex = elements.cols();
    // Eigen::VectorXd u = Eigen::VectorXd::Zero(2*numVertex);

    // initialize consistency and stability matrices
    Eigen::MatrixXd Kc = Eigen::MatrixXd::Zero(2*numVertex, 2*numVertex);
    Eigen::MatrixXd Ks = Eigen::MatrixXd::Zero(2*numVertex, 2*numVertex);

    // geometric information
    double area;
    Eigen::VectorXd Lengths = Eigen::VectorXd::Zero(numVertex);
    Eigen::MatrixXd normal = Eigen::MatrixXd::Zero(numVertex, 2);
    
    // compute geometric information
    utils::operations op;
    area = op.calcArea(coords);

    for(int i=0; i<numVertex; i++){
        Eigen::MatrixXd startNode = coords.row(i);
        Eigen::MatrixXd endNode = coords.row((i+1)%numVertex);
        // std::cout << "endNode :: " << (i+1)%numVertex << std::endl;

        Eigen::MatrixXd edge = op.buildEdge(startNode, endNode);
        Lengths(i) = op.calcLength(edge);
        // std::cout << "Lengths :: " << Lengths(i) << std::endl;
        normal.row(i) = op.computerNormalVector(edge);
        // std::cout << "normal :: " << normal.row(i) << std::endl;
    }

    switch(numVertex) {
        case 3:
            localKcTriangle(Lengths, &area, &Mu, &La, u, normal, Kc);
            op.forceSymmetry(Kc);
            // std::cout << "Local Stiffness Matrix" << std::endl;
            // std::cout << Kc << "\n" << std::endl;
            break;
        case 4:
            localKcQuadrilateral(Lengths, &area, &Mu, &La, u, normal, Kc);
            op.forceSymmetry(Kc);
            // std::cout << "Local Stiffness Matrix" << std::endl;
            // std::cout << Kc << std::endl;
            break;
        default:
            std::cout << "Only  triangular and quadrilateral elements are supported at the moment." << std::endl;
            break;
    }


    return Kc + Ks;
};


Eigen::MatrixXd solver::nonlinear2d::buildGlobalK(Eigen::VectorXd u){
    int ndof = 2*nodes.rows();
    int ne = elements.rows();

    // std::cout << "Number of dofs: " << ndof << std::endl;
    // std::cout << "Number of elements: " << ne << "\n" << std::endl;

    // stiffness matrices
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(ndof,ndof);
    Eigen::MatrixXd Kloc;
    
    // displacements, elements and coordinates
    Eigen::VectorXd ue;
    Eigen::MatrixXi e; // store individual element for calculations
    Eigen::VectorXi e_dofs;
    Eigen::MatrixXd coords;

    utils::operations op;

    for(int i=0; i<ne; i++){
        e = elements.row(i);
        e_dofs = op.getOrder1Indices(e);
        coords = op.getCoordinatesPlane(e, nodes);
        ue = op.getVectorValues(e, u);
        Kloc = buildLocalK(coords, ue);
        K = op.assembleMatrix(K, Kloc, e_dofs);
    }

    // std::cout << "Global stiffness matrix assembled" << std::endl;
    // std::cout << K << "\n" << std::endl;

    return K;
};

Eigen::VectorXd solver::nonlinear2d::buildGlobalR(Eigen::VectorXd u){
    int ndof = 2*nodes.rows();
    int ne = elements.rows();

    Eigen::VectorXd R = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd R_loc;

    // displacements, elements and coordinates
    Eigen::VectorXd ue;
    Eigen::MatrixXi e; // store individual element for calculations
    Eigen::VectorXi e_dofs;
    Eigen::MatrixXd coords;

    utils::operations op;

    for(int i=0; i<ne; i++){
        e = elements.row(i);
        e_dofs = op.getOrder1Indices(e);
        coords = op.getCoordinatesPlane(e, nodes);
        ue = op.getVectorValues(e, u);
        R_loc = buildLocalRc(coords, ue);
        R = op.assembleVector(R, R_loc, e_dofs);
    }

    // std::cout << "Global residual vector assembled" << std::endl;
    // std::cout << R << "\n" << std::endl;

    return R;
}

Eigen::MatrixXd solver::nonlinear2d::applyDBC(Eigen::MatrixXd K){
    int n_supp = supp.rows();
    int index; // dof index

    for(int i=0; i <n_supp; i++){
        if(supp(i,1)==1){
            index = 2*supp(i,0);
            K.row(index).setZero();
            K.col(index).setZero();
            K(index,index) = 1;
        }

        if(supp(i,2)==1){
            index = 2*supp(i,0)+1;
            K.row(index).setZero();
            K.col(index).setZero();
            K(index,index) = 1;
        }
        
    }
    return K;
 };

// Eigen::VectorXd solver::nonlinear2d::applyDBCVector(Eigen::VectorXd R){
//     // ... implementation ...
// }

Eigen::VectorXd solver::nonlinear2d::applyNBC(Eigen::VectorXd u){
    int n = load.rows();
    int ndof = 2*nodes.rows();
    int startIndex, endIndex;
    double length;

    Eigen::VectorXd Rt = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd Rt_loc;
    Eigen::MatrixXd startNode, endNode;
    Eigen::VectorXi e_dofs;
    Eigen::MatrixXi e;
    Eigen::MatrixXd edge;
    Eigen::VectorXd ue;

    utils::operations op;

    for(int i=0; i<n; i++){
        e = load.row(i);
        startIndex = load(i,0);
        endIndex = load(i,1);
        startNode = nodes(startIndex, Eigen::all);
        endNode = nodes(endIndex, Eigen::all);
        edge = op.buildEdge(startNode, endNode);
        length = op.calcLength(edge);
        ue = op.getVectorValues(e, u);
        Rt_loc = buildLocalLoadVector(length, qx/n, qy/n, ue);
        e_dofs = op.getOrder1Indices(e);
        Rt = op.assembleVector(Rt, Rt_loc, e_dofs);
    }

    // std::cout << "Global load vector assembled" << std::endl;
    // std::cout << Rt << "\n" << std::endl;

    return Rt;
 };

 Eigen::VectorXd solver::nonlinear2d::newtonRaphson(Eigen::VectorXd u0, double qx0, double qy0){
    int ndof = 2*nodes.rows();
    int maxIterations = 500;
    double tolerance = 1e-8;
    double alpha = 0.5;

    Eigen::VectorXd u = u0;
    Eigen::VectorXd delta_u = Eigen::VectorXd::Zero(ndof);

    Eigen::VectorXd R = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd Re = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd Rt = Eigen::VectorXd::Zero(ndof);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(ndof,ndof);

    K = buildGlobalK(u);
    K = applyDBC(K);

    for(int iter=0; iter<maxIterations; iter++){

        std::cout << "Iter: " << iter << std::endl;

        // STEP 1: calculate residual and stiffness matrix for u_(i)
        Re = buildGlobalR(u);
        Rt = applyNBC(u);
        R = Re + Rt;
        
        R = applyDBCVector(R);

        // STEP 2: calculate the displacement increment
        delta_u = K.ldlt().solve(-R);

        // STEP 3: update the displacement
        u = u + alpha * delta_u;

        std::cout<< "Max u: " << u.maxCoeff() <<std::endl; 

        // STEP 4: compute the residual vector for u_(i+1)
        Re = buildGlobalR(u);
        Rt = applyNBC(u);
        R = Re + Rt;
        R = applyDBCVector(R);
        std::cout << "Residual: " << R.norm() << std::endl;
        // std::cout << R <<  std::endl;

        // STEP 5: check for convergence
        if(R.norm() < tolerance){
            std::cout << "Convergence achieved after " << iter << " iterations." << std::endl;
            break;
        };
    }

    return u;
 };

 Eigen::VectorXd solver::nonlinear2d::solve(Eigen::VectorXd u0, double load_step){
    double qx0 = load_step;
    utils::logging log;

    try
    {
        while(qx0<qx){
            std::cout << "Load step: " << qx0 << std::endl;
            u0 = newtonRaphson(u0, qx0, 0.0);
            qx0 += load_step;
        };

        std::map<std::string, std::string> dataMap = {
            {"solver", "nonlinear2d"},
            {"load_step", std::to_string(load_step)},
            {"ndof", std::to_string(2*nodes.rows())},
            {"ne", std::to_string(elements.rows())},
            {"qx", std::to_string(qx)},
            {"qy", std::to_string(qy)},
            {"La", std::to_string(La)},
            {"Mu", std::to_string(Mu)},
            {"u0_max", std::to_string(u0.maxCoeff())},
            {"status", "success"}
        };
        log.buildLogFile(dataMap);
    }
    catch(const std::exception& e)
    {
        std::map<std::string, std::string> dataMap = {
            {"solver", "nonlinear2d"},
            {"status", "failed"},
            {"info", e.what()}
        };
        log.buildLogFile(dataMap);
        std::cerr << e.what() << '\n';
    }
    
    return u0;

 }

