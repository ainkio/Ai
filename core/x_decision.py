"""
x_decision.py - 决策模块（重构版）
基于胜率、赔率、OUT数的数学化决策系统
作者：AI重构
日期：2024
"""

from treys import Card, Evaluator, Deck
import random
from collections import Counter
from core.comm import *
from core.x_yolo import detect_player_count
import math


class PokerDecisionMaker:
    """德州扑克决策器 - 基于数学计算的理性决策"""
    
    def __init__(self):
        """初始化决策器"""
        self.evaluator = Evaluator()
        BoxText("成功：决策器初始化完成")
    
    def make_decision(self, vision_data):
        """
        做出决策
        
        Args:
            vision_data: 从 x_vision.py 获取的视觉信息
            
        Returns:
            dict: {
                "action": str,  # 决策动作：fold/check/call/raise
                "amount": float,  # 下注金额
                "reason": str,  # 决策理由
                "win_rate": float,  # 胜率
                "pot_odds": float,  # 底池赔率
                "outs": int,  # OUT数（成牌的出路数）
                "display_text": str  # 显示文本
            }
        """
        try:
            # 1. 提取关键信息
            hand_cards = vision_data.get("hand_cards", [])
            public_cards = vision_data.get("public_cards", [])
            pot_chips = float(vision_data.get("pot_chips", 1.0))
            my_chips = float(vision_data.get("my_chips", 10.0))
            bet_amount = float(vision_data.get("bet_amount", 0.0))
            blind = float(vision_data.get("blind", 0.1))
            my_position = vision_data.get("my_position", "unknown")
            players_after_me = int(vision_data.get("players_after_me", 0))
            game_stage = vision_data.get("game_stage", "preflop")
            
            # 验证和修正底池金额（防止识别错误）
            if bet_amount > 0:
                # 底池应该 >= 下注金额（否则赔率就不合理）
                if pot_chips < bet_amount:
                    BoxText(f"警告：底池金额{pot_chips}小于下注金额{bet_amount}，可能识别错误，使用修正值")
                    pot_chips = max(bet_amount * 2, blind * 3)  # 使用合理的最小值
                
                # 检查赔率是否合理（底池赔率通常在1.5:1到10:1之间）
                preliminary_pot_odds = (pot_chips + bet_amount) / bet_amount if bet_amount > 0 else float('inf')
                if preliminary_pot_odds > 20:
                    BoxText(f"警告：底池赔率{preliminary_pot_odds:.1f}:1过大，可能底池识别错误")
                    # 假设合理的赔率应该是3:1左右
                    pot_chips = bet_amount * 2  # 修正为合理值
            
            # ======== 使用YOLO检测对手数量（快速、准确）========
            opponent_count = detect_player_count("temp/real_img.png")
            
            # 检查YOLO检测结果
            if opponent_count <= 0:
                BoxText(f"⚠️ 警告：YOLO检测失败，使用默认值1")
                opponent_count = 1
            
            BoxText(f"✅ 对手数量：{opponent_count}（由YOLO提供）")
            
            # 2. 验证手牌
            if not hand_cards or len(hand_cards) != 2:
                return self._error_decision("手牌信息不足")
            
            # 3. 计算胜率
            win_rate = self._calculate_win_rate(hand_cards, public_cards, opponent_count)
            
            # 4. 计算OUT数（成牌的出路数）
            outs = self._calculate_outs(hand_cards, public_cards)
            
            # 5. 计算底池赔率
            pot_odds = self._calculate_pot_odds(pot_chips, bet_amount)
            
            # 6. 计算隐含赔率（Implied Odds）
            implied_odds = self._calculate_implied_odds(pot_chips, bet_amount, my_chips)
            
            # 7. 基于数学模型做决策
            decision = self._make_mathematical_decision(
                win_rate=win_rate,
                outs=outs,
                pot_odds=pot_odds,
                implied_odds=implied_odds,
                pot_chips=pot_chips,
                my_chips=my_chips,
                bet_amount=bet_amount,
                blind=blind,
                my_position=my_position,
                players_after_me=players_after_me,
                game_stage=game_stage,
                hand_cards=hand_cards,
                public_cards=public_cards
            )
            
            # 8. 构建显示文本
            display_text = self._build_display_text(
                decision, win_rate, outs, pot_odds, 
                hand_cards, public_cards, pot_chips, my_chips, bet_amount
            )
            
            decision["display_text"] = display_text
            decision["win_rate"] = win_rate
            decision["pot_odds"] = pot_odds
            decision["outs"] = outs
            
            # 9. 输出详细的决策日志
            self._log_decision_details(
                hand_cards, public_cards, win_rate, outs, pot_odds, implied_odds,
                pot_chips, my_chips, bet_amount, opponent_count, my_position,
                game_stage, decision
            )
            
            return decision
            
        except Exception as e:
            BoxText(f"错误：决策失败 - {str(e)}")
            return self._error_decision(f"决策异常: {str(e)}")
    
    def _calculate_win_rate(self, hand_cards, public_cards, opponent_count, simulations=1000):
        """
        蒙特卡洛模拟计算胜率
        
        Args:
            hand_cards: 手牌列表 ['As', 'Kd']
            public_cards: 公牌列表 ['Qs', 'Jh', 'Tc']
            opponent_count: 对手数量
            simulations: 模拟次数
            
        Returns:
            float: 胜率百分比（0-100）
        """
        try:
            # 标准化牌面格式（确保点数大写）
            hand_cards = [self._normalize_card_format(card) for card in hand_cards]
            public_cards = [self._normalize_card_format(card) for card in public_cards] if public_cards else []
            
            deck = Deck()
            hand = [Card.new(card) for card in hand_cards]
            public = [Card.new(card) for card in public_cards] if public_cards else []
            
            # 从牌堆中移除已知牌
            remaining = deck.cards[:]
            for card in hand + public:
                if card in remaining:
                    remaining.remove(card)
            
            cards_needed = 5 - len(public)  # 还需要几张公牌
            opp_cards_needed = opponent_count * 2  # 对手需要的手牌数
            
            wins = 0
            valid_simulations = 0
            for _ in range(simulations):
                if len(remaining) < (cards_needed + opp_cards_needed):
                    continue
                
                # 随机抽取公牌和对手手牌
                sampled_cards = random.sample(remaining, cards_needed + opp_cards_needed)
                full_public = public + sampled_cards[:cards_needed]
                
                # 对手手牌
                opp_hands = [
                    sampled_cards[cards_needed + i*2:cards_needed + (i+1)*2] 
                    for i in range(opponent_count)
                ]
                
                # 评估牌力
                my_rank = self.evaluator.evaluate(full_public, hand)
                best_opp_rank = min([
                    self.evaluator.evaluate(full_public, opp_hand) 
                    for opp_hand in opp_hands
                ]) if opp_hands else float('inf')
                
                if my_rank < best_opp_rank:
                    wins += 1
                elif my_rank == best_opp_rank:
                    wins += 0.5
                
                valid_simulations += 1
            
            win_rate = round((wins / valid_simulations) * 100, 2) if valid_simulations > 0 else 0
            
            # 输出调试信息（仅在翻牌圈第一次）
            if len(public) == 0 and win_rate > 0:
                BoxText(f"调试：胜率计算完成 - 手牌{hand_cards}, 对手{opponent_count}个, 模拟{valid_simulations}次, 胜率{win_rate}%")
            
            return win_rate
            
        except Exception as e:
            import traceback
            BoxText(f"警告：胜率计算失败 - {str(e)}")
            BoxText(f"调试：手牌={hand_cards}, 公牌={public_cards}, 对手={opponent_count}")
            traceback.print_exc()
            return 30.0  # 返回保守默认值
    
    def _calculate_outs(self, hand_cards, public_cards):
        """
        计算OUT数（成牌的出路数）
        
        Args:
            hand_cards: 手牌
            public_cards: 公牌
            
        Returns:
            int: OUT数
        """
        try:
            if not public_cards or len(public_cards) < 3:
                return 0  # 翻牌前无法计算OUT
            
            # 标准化牌面格式
            hand_cards = [self._normalize_card_format(card) for card in hand_cards]
            public_cards = [self._normalize_card_format(card) for card in public_cards]
            
            # 提取花色和点数
            RANKS = "23456789TJQKA"
            SUITS = "shdc"
            
            hand_ranks = [RANKS.index(card[0].upper()) for card in hand_cards if len(card) >= 2]
            hand_suits = [SUITS.index(card[1].lower()) for card in hand_cards if len(card) >= 2]
            public_ranks = [RANKS.index(card[0].upper()) for card in public_cards if len(card) >= 2]
            public_suits = [SUITS.index(card[1].lower()) for card in public_cards if len(card) >= 2]
            
            all_ranks = hand_ranks + public_ranks
            all_suits = hand_suits + public_suits
            
            outs = 0
            
            # 1. 同花听牌（Flush Draw）- 4张同花色
            suit_counts = Counter(all_suits)
            for suit, count in suit_counts.items():
                if count == 4:
                    outs += 9  # 还有9张该花色的牌
            
            # 2. 顺子听牌（Straight Draw）
            unique_ranks = sorted(set(all_ranks))
            
            # 两头顺（Open-ended Straight Draw）
            for i in range(len(unique_ranks) - 3):
                window = unique_ranks[i:i+4]
                if window[-1] - window[0] == 3:  # 连续4张
                    outs += 8  # 两头顺有8个OUT
                    break
            
            # 卡顺（Inside Straight Draw）
            if outs == 0:  # 如果不是两头顺，检查卡顺
                for i in range(len(unique_ranks) - 2):
                    if unique_ranks[i+2] - unique_ranks[i] == 4 and unique_ranks[i+1] - unique_ranks[i] != 1:
                        outs += 4  # 卡顺有4个OUT
                        break
            
            # 3. 对子听牌（Pair Draw）- 可以成三条
            rank_counts = Counter(all_ranks)
            for rank, count in rank_counts.items():
                if count == 2 and rank in hand_ranks:
                    outs += 2  # 有2个OUT可以成三条
            
            # 4. 两对听牌（Two Pair Draw）- 可以成葫芦
            pairs = [rank for rank, count in rank_counts.items() if count == 2]
            if len(pairs) >= 2:
                outs += 4  # 两对可以成葫芦
            
            return outs
            
        except Exception as e:
            BoxText(f"警告：OUT计算失败 - {str(e)}")
            return 0
    
    def _calculate_pot_odds(self, pot_chips, bet_amount):
        """
        计算底池赔率
        
        Args:
            pot_chips: 底池金额
            bet_amount: 需要跟注的金额
            
        Returns:
            float: 底池赔率（比如 3.5 表示 3.5:1）
        """
        if bet_amount == 0:
            return float('inf')  # 免费看牌
        
        total_pot = pot_chips + bet_amount
        pot_odds = round(total_pot / bet_amount, 2)
        return pot_odds
    
    def _calculate_implied_odds(self, pot_chips, bet_amount, my_chips):
        """
        计算隐含赔率（考虑后续可能赢得的筹码）
        
        Args:
            pot_chips: 底池金额
            bet_amount: 需要跟注的金额
            my_chips: 我的筹码
            
        Returns:
            float: 隐含赔率
        """
        if bet_amount == 0:
            return float('inf')
        
        # 假设成牌后可以额外赢得对手 50% 的剩余筹码
        potential_winnings = pot_chips + bet_amount + (my_chips * 0.5)
        implied_odds = round(potential_winnings / bet_amount, 2)
        return implied_odds
    
    def _make_mathematical_decision(self, win_rate, outs, pot_odds, implied_odds, 
                                    pot_chips, my_chips, bet_amount, blind, 
                                    my_position, players_after_me, game_stage,
                                    hand_cards, public_cards):
        """
        基于数学模型做决策
        
        核心原则：
        1. 如果有利可图（胜率 > 所需胜率），则跟进
        2. 如果不划算，则弃牌
        3. 下注金额根据牌力和位置动态调整
        
        Returns:
            dict: 决策结果
        """
        
        # === 特殊情况处理 ===
        
        # 1. 免费看牌（无需下注）
        if bet_amount == 0:
            # 如果牌力强，考虑下注
            if win_rate >= 75:
                # 强牌下注（底池的 2/3）
                raise_amount = round(pot_chips * 0.67, 2)
                return {
                    "action": "raise",
                    "amount": raise_amount,
                    "reason": f"强牌价值下注（胜率{win_rate}%）"
                }
            elif win_rate >= 60:
                # 中等牌下注（底池的 1/2）
                raise_amount = round(pot_chips * 0.5, 2)
                return {
                    "action": "raise",
                    "amount": raise_amount,
                    "reason": f"中等牌试探性下注（胜率{win_rate}%）"
                }
            else:
                # 弱牌免费看牌
                return {
                    "action": "check",
                    "amount": 0,
                    "reason": f"免费看牌（胜率{win_rate}%，等待改进）"
                }
        
        # 2. 全下情况（下注金额 >= 我的筹码）
        if bet_amount >= my_chips:
            # 计算所需胜率
            required_win_rate = (bet_amount / (pot_chips + bet_amount)) * 100
            if win_rate >= required_win_rate:
                return {
                    "action": "call",
                    "amount": my_chips,
                    "reason": f"全下跟注（胜率{win_rate}% > 所需{required_win_rate:.1f}%）"
                }
            else:
                return {
                    "action": "fold",
                    "amount": 0,
                    "reason": f"弃牌保护筹码（胜率{win_rate}% < 所需{required_win_rate:.1f}%）"
                }
        
        # === 核心决策逻辑：基于期望值（EV）===
        
        # 计算所需胜率（考虑底池赔率）
        required_win_rate = (bet_amount / (pot_chips + bet_amount)) * 100
        
        # 如果有OUT，使用OUT计算期望胜率
        if outs > 0 and len(public_cards) < 5:
            # 转牌圈或河牌圈
            cards_to_come = 2 if len(public_cards) == 3 else 1
            # 粗略计算：每个OUT约有2%的概率（河牌）或4%（转牌+河牌）
            out_win_rate = outs * (4 if cards_to_come == 2 else 2)
            out_win_rate = min(out_win_rate, 100)  # 不超过100%
            
            # 综合胜率（当前胜率 + OUT胜率）
            combined_win_rate = min(win_rate + out_win_rate * 0.5, 100)
        else:
            combined_win_rate = win_rate
        
        # === 决策判断 ===
        
        # 1. 极强牌（胜率 >= 80%）- 价值最大化
        if combined_win_rate >= 80:
            if bet_amount <= pot_chips * 0.3:
                # 对手下注很小，加注
                raise_amount = round(pot_chips * 1.0, 2)
                return {
                    "action": "raise",
                    "amount": raise_amount,
                    "reason": f"极强牌加注（胜率{combined_win_rate:.1f}%，底池1倍）"
                }
            else:
                # 对手下注较大，跟注
                return {
                    "action": "call",
                    "amount": bet_amount,
                    "reason": f"极强牌跟注（胜率{combined_win_rate:.1f}%）"
                }
        
        # 2. 强牌（胜率 60-80%）
        elif combined_win_rate >= 60:
            if bet_amount <= pot_chips * 0.5:
                # 对手下注适中，加注
                raise_amount = round(pot_chips * 0.75, 2)
                return {
                    "action": "raise",
                    "amount": raise_amount,
                    "reason": f"强牌加注（胜率{combined_win_rate:.1f}%，底池3/4倍）"
                }
            else:
                # 对手下注较大，跟注
                return {
                    "action": "call",
                    "amount": bet_amount,
                    "reason": f"强牌跟注（胜率{combined_win_rate:.1f}%）"
                }
        
        # 3. 中等牌（胜率 45-60%）
        elif combined_win_rate >= 45:
            # 提高安全边际到10%
            if combined_win_rate >= required_win_rate + 10:  # 留有10%的安全边际
                return {
                    "action": "call",
                    "amount": bet_amount,
                    "reason": f"中等牌跟注（胜率{combined_win_rate:.1f}% > 所需{required_win_rate:.1f}%+10%）"
                }
            else:
                return {
                    "action": "fold",
                    "amount": 0,
                    "reason": f"中等牌弃牌（胜率{combined_win_rate:.1f}% 安全边际不足）"
                }
        
        # 4. 听牌（有OUT但当前胜率不高）
        elif outs >= 8:  # 同花+顺子等强听牌
            # 检查隐含赔率是否足够
            if implied_odds >= 4.0:  # 隐含赔率够好
                return {
                    "action": "call",
                    "amount": bet_amount,
                    "reason": f"强听牌跟注（{outs}个OUT，隐含赔率{implied_odds:.1f}:1）"
                }
            elif pot_odds >= 4.0:  # 底池赔率够好
                return {
                    "action": "call",
                    "amount": bet_amount,
                    "reason": f"强听牌跟注（{outs}个OUT，底池赔率{pot_odds:.1f}:1）"
                }
            else:
                return {
                    "action": "fold",
                    "amount": 0,
                    "reason": f"听牌弃牌（{outs}个OUT，赔率不足）"
                }
        
        elif outs >= 4:  # 一般听牌
            if pot_odds >= 5.0:
                return {
                    "action": "call",
                    "amount": bet_amount,
                    "reason": f"听牌跟注（{outs}个OUT，底池赔率{pot_odds:.1f}:1）"
                }
            else:
                return {
                    "action": "fold",
                    "amount": 0,
                    "reason": f"听牌弃牌（{outs}个OUT，赔率不足）"
                }
        
        # 5. 弱牌（胜率 < 45%，无听牌）
        else:
            # 弱牌要求更高的安全边际（15%）
            if combined_win_rate >= required_win_rate + 15:
                # 虽然牌弱，但赔率非常好
                return {
                    "action": "call",
                    "amount": bet_amount,
                    "reason": f"赔率极佳跟注（胜率{combined_win_rate:.1f}% > 所需{required_win_rate:.1f}%+15%）"
                }
            else:
                # 弃牌
                return {
                    "action": "fold",
                    "amount": 0,
                    "reason": f"弱牌弃牌（胜率{combined_win_rate:.1f}%过低，不值得跟注）"
                }
    
    def _normalize_card_format(self, card):
        """
        标准化牌面格式
        
        Args:
            card: 牌面字符串，如 'ks', 'Kd', 'qh', '10s', 'tc' 等
            
        Returns:
            str: 标准化后的牌面，如 'Ks', 'Kd', 'Qh', 'Ts' 等（点数大写+花色小写）
        """
        if not card or len(card) < 2:
            return card
        
        # 提取点数和花色
        rank = card[:-1].upper()  # 点数转大写
        suit = card[-1].lower()   # 花色转小写
        
        # 10转换为T
        if rank == '10':
            rank = 'T'
        
        # 验证格式
        valid_ranks = '23456789TJQKA'
        valid_suits = 'shdc'
        
        if rank in valid_ranks and suit in valid_suits:
            return f"{rank}{suit}"
        
        BoxText(f"警告：无效的牌面格式 '{card}'，已忽略")
        return card
    
    def _build_display_text(self, decision, win_rate, outs, pot_odds,
                           hand_cards, public_cards, pot_chips, my_chips, bet_amount):
        """构建显示文本"""
        
        # 格式化牌面
        def format_card(card):
            suits = {'s': '♠', 'h': '♥', 'c': '♣', 'd': '♦'}
            card = self._normalize_card_format(card)  # 标准化格式
            if len(card) >= 2:
                rank = card[:-1].upper()
                suit = card[-1].lower()
                return f"{rank}{suits.get(suit, '')}"
            return card
        
        hand_str = "".join([format_card(c) for c in hand_cards])
        public_str = "".join([format_card(c) for c in public_cards]) if public_cards else "无"
        
        lines = [
            f"行动：{decision['action']}",
            f"金额：{decision.get('amount', 0)}",
            f"理由：{decision['reason']}",
            "",
            f"手牌：{hand_str}",
            f"公牌：{public_str}",
            f"胜率：{win_rate}%",
            f"OUT数：{outs}",
            f"底池：{pot_chips}",
            f"筹码：{my_chips}",
            f"下注：{bet_amount}",
            f"赔率：{pot_odds}:1"
        ]
        
        return "\n".join(lines)
    
    def _log_decision_details(self, hand_cards, public_cards, win_rate, outs, 
                              pot_odds, implied_odds, pot_chips, my_chips, bet_amount,
                              opponent_count, my_position, game_stage, decision):
        """
        输出详细的决策日志
        """
        # 格式化牌面
        def format_card(card):
            suits = {'s': '♠', 'h': '♥', 'c': '♣', 'd': '♦'}
            card = self._normalize_card_format(card)
            if len(card) >= 2:
                rank = card[:-1].upper()
                suit = card[-1].lower()
                return f"{rank}{suits.get(suit, '')}"
            return card
        
        hand_str = "".join([format_card(c) for c in hand_cards])
        public_str = "".join([format_card(c) for c in public_cards]) if public_cards else "无"
        
        # 计算所需胜率
        if bet_amount > 0:
            required_win_rate = (bet_amount / (pot_chips + bet_amount)) * 100
        else:
            required_win_rate = 0
        
        # 构建日志
        BoxText("=" * 60)
        BoxText(f"决策详情：{game_stage}")
        BoxText("-" * 60)
        BoxText(f"牌面信息：手牌 {hand_str} | 公牌 {public_str}")
        BoxText(f"筹码信息：底池 {pot_chips} | 我的筹码 {my_chips} | 需下注 {bet_amount}")
        BoxText(f"位置信息：我的位置 {my_position} | 对手数量 {opponent_count}")
        BoxText("-" * 60)
        BoxText(f"牌力评估：胜率 {win_rate}% | OUT数 {outs}个")
        BoxText(f"赔率分析：底池赔率 {pot_odds}:1 | 隐含赔率 {implied_odds}:1 | 所需胜率 {required_win_rate:.1f}%")
        BoxText("-" * 60)
        BoxText(f"决策结论：{decision['action'].upper()} {decision.get('amount', 0)}")
        BoxText(f"决策理由：{decision['reason']}")
        BoxText("=" * 60)
    
    def _error_decision(self, reason):
        """返回错误决策"""
        return {
            "action": "fold",
            "amount": 0,
            "reason": reason,
            "win_rate": 0,
            "pot_odds": 0,
            "outs": 0,
            "display_text": f"错误：{reason}\n建议弃牌"
        }


# 全局单例
_decision_maker = None

def make_decision(vision_data):
    """
    全局函数 - 做出决策
    
    Args:
        vision_data: 视觉信息
        
    Returns:
        dict: 决策结果
    """
    global _decision_maker
    if _decision_maker is None:
        _decision_maker = PokerDecisionMaker()
    return _decision_maker.make_decision(vision_data)


# 测试代码
if __name__ == "__main__":
    # 测试决策功能
    test_data = {
        "hand_cards": ["Ks", "6c"],
        "public_cards": ["Js", "3d", "Th"],
        "pot_chips": 18.6,
        "my_chips": 18.5,
        "bet_amount": 9.3,
        "blind": 0.1,
        "my_position": "middle",
        "players_after_me": 4,
        "game_stage": "flop",
        "players": [
            {"name": "玩家1", "action": "raise", "chips": 60, "is_dealer": true, "has_folded": false},
            {"name": "玩家2", "action": "fold", "chips": 0, "is_dealer": false, "has_folded": true},
            {"name": "我", "action": "waiting", "chips": 18.5, "is_dealer": false, "has_folded": false},
            {"name": "玩家3", "action": "call", "chips": 50, "is_dealer": false, "has_folded": false}
        ]
    }
    
    result = make_decision(test_data)
    print(f"决策结果：{result['action']} {result.get('amount', 0)}")
    print(f"理由：{result['reason']}")
    print(f"\n{result['display_text']}")

