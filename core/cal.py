from treys import Card, Evaluator, Deck
import random
from collections import Counter
from core.comm import *
from core.yolo import *
from core.ai import *
import math
import threading

# 语音播报包装函数
def speak_decision(text):
    """在新线程中播报决策"""
    BoxText(f"语音：准备播报 '{text}'")
    def run_audio():
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            try:
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Volume = 100
                BoxText(f"语音：正在播报 '{text}'")
                speaker.Speak(text)
                BoxText(f"语音：播报完成 '{text}'")
            finally:
                pythoncom.CoUninitialize()
        except Exception as e:
            BoxText(f"语音：播报失败 '{text}' - {e}")
    
    thread = threading.Thread(target=run_audio, daemon=True)
    thread.start()

def MA_MM(hd, pc, pr, simulations=1000):
    evaluator, deck, wins = Evaluator(), Deck(), 0
    hand_cards, public_cards = [Card.new(card) for card in hd], [Card.new(card) for card in pc]
    remaining = deck.cards[:]
    [remaining.remove(card) for card in hand_cards + public_cards if card in remaining]
    cards_needed, opp_cards_needed = 5 - len(pc), pr * 2
    for _ in range(simulations):
        if len(remaining) < (cards_needed + opp_cards_needed): continue
        sampled_cards = random.sample(remaining, cards_needed + opp_cards_needed)
        full_public = public_cards + sampled_cards[:cards_needed]
        opp_hands = [sampled_cards[cards_needed + i * 2:cards_needed + (i + 1) * 2] for i in range(pr)]
        my_rank = evaluator.evaluate(full_public, hand_cards)
        best_opp_rank = min([evaluator.evaluate(full_public, opp_hand) for opp_hand in opp_hands]) if opp_hands else float('inf')
        wins += 1 if my_rank < best_opp_rank else 0.5 if my_rank == best_opp_rank else 0
    return round((wins / simulations) * 100 if simulations > 0 else 0, 2)

def MA_PL(hd, pc):
    simulations = 1000
    evaluator, deck, total_rank = Evaluator(), Deck(), 0
    hand_cards, public_cards = [Card.new(card) for card in hd], [Card.new(card) for card in pc]
    remaining = deck.cards[:]
    [remaining.remove(card) for card in hand_cards + public_cards if card in remaining]
    cards_needed = 5 - len(pc)
    for _ in range(simulations):
        if len(remaining) < cards_needed: continue
        sampled_public = public_cards + random.sample(remaining, cards_needed)
        total_rank += evaluator.evaluate(sampled_public, hand_cards)
    avg_rank = total_rank / simulations if simulations > 0 else 7555
    # return round(max(50, min(100, int(100 * (1 - (avg_rank - 1) / (7555 - 1)) + 20))), 2)
    return avg_rank, round(100 * (1 - (avg_rank - 1) / (7555 - 1)), 2)

def ML_GP(hd, pc):
    if len(pc) < 3: return "暂无公牌"

    RANKS = "23456789TJQKA"
    SUITS = "shdc"

    hand_suits = [SUITS.index(card[1]) for card in hd]
    hand_ranks = [RANKS.index(card[0]) for card in hd]
    public_suits = [SUITS.index(card[1]) for card in pc]
    public_ranks = sorted([RANKS.index(card[0]) for card in pc])

    effect, strength = "无影响", "neutral"

    # 河牌圈（5张公牌）特殊处理
    if len(pc) == 5:
        all_ranks = sorted(set(public_ranks + hand_ranks))
        all_suits = public_suits + hand_suits
        all_cards_by_suit = {suit: sorted([r for r, s in zip(public_ranks + hand_ranks, all_suits) if s == suit]) for suit in set(all_suits)}

        # 检查顺子
        is_straight = False
        if len(all_ranks) >= 5:
            for i in range(len(all_ranks) - 4):
                if all_ranks[i + 4] - all_ranks[i] == 4:
                    is_straight = True
                    effect = "顺子"
                    break
            if not is_straight and 12 in all_ranks and 0 in all_ranks and 1 in all_ranks and 2 in all_ranks:
                is_straight = True
                effect = "顺子"

        # 检查同花
        is_flush = False
        for suit, ranks in all_cards_by_suit.items():
            if len(ranks) >= 5:
                is_flush = True
                effect = "同花" if not is_straight else "同花顺"
                break

        # 如果成牌，直接返回牌型
        if is_straight or is_flush:
            return effect

        # 未成牌，评估影响
        if any(count >= 2 for count in Counter(public_ranks).values()):
            if any(rank in hand_ranks for rank, count in Counter(public_ranks).items() if count >= 2):
                return "影响正面"  # 能凑成三条或更好
            return "影响负面"  # 公牌有对子但手牌无法利用
        return "影响中性"  # 无特殊牌型或影响

    # 非河牌圈（3或4张公牌）的原有逻辑
    all_ranks = sorted(set(public_ranks + hand_ranks))
    all_suits = public_suits + hand_suits
    all_cards_by_suit = {suit: sorted([r for r, s in zip(public_ranks + hand_ranks, all_suits) if s == suit]) for suit in set(all_suits)}

    # 检查是否已成牌
    is_straight = False
    if len(all_ranks) >= 5:
        for i in range(len(all_ranks) - 4):
            if all_ranks[i + 4] - all_ranks[i] == 4:
                is_straight = True
                effect = "顺子"
                strength = "strong_positive"
                break
        if not is_straight and 12 in all_ranks and 0 in all_ranks and 1 in all_ranks and 2 in all_ranks:
            is_straight = True
            effect = "顺子"
            strength = "strong_positive"

    is_flush = False
    for suit, ranks in all_cards_by_suit.items():
        if len(ranks) >= 5:
            is_flush = True
            effect = "同花" if not is_straight else "同花顺"
            strength = "strong_positive" if not is_straight else "super_positive"
            break

    if is_straight or is_flush:
        return effect if strength == "neutral" else f"{effect}，影响{'超强正面' if 'super' in strength else '正面'}"

    # 未成牌时检查潜力
    suit_counts = Counter(public_suits)
    dominant_suit, suit_count = max(suit_counts.items(), key=lambda x: x[1], default=(None, 0))
    flush_potential = suit_count >= 3

    straight_type, has_straight = None, False
    if len(all_ranks) >= 4:
        for i in range(len(all_ranks) - 3):
            window = all_ranks[i:i + 4]
            gaps = [window[j + 1] - window[j] for j in range(3)]
            if gaps.count(1) == 2 and gaps.count(2) == 1 and (gaps[0] == 2 or gaps[2] == 2):
                straight_type = "inside"
                has_straight = True
                break
            elif gaps.count(1) == 2 and gaps.count(2) == 1 and gaps[1] == 2:
                straight_type = "inside"
                has_straight = True
                break
            elif gaps == [1, 1, 1]:
                straight_type = "double_ended"
                has_straight = True
                break
        if not has_straight and 12 in all_ranks and 0 in all_ranks and 1 in all_ranks and not (2 in all_ranks and 3 in all_ranks):
            straight_type = "inside"
            has_straight = True

    public_straight_type, public_has_straight = None, False
    if len(public_ranks) >= 4:
        for i in range(len(public_ranks) - 3):
            window = public_ranks[i:i + 4]
            gaps = [window[j + 1] - window[j] for j in range(3)]
            if gaps.count(1) == 2 and gaps.count(2) == 1 and (gaps[0] == 2 or gaps[2] == 2):
                public_straight_type = "inside"
                public_has_straight = True
                break
            elif gaps.count(1) == 2 and gaps.count(2) == 1 and gaps[1] == 2:
                public_straight_type = "inside"
                public_has_straight = True
                break
            elif gaps == [1, 1, 1]:
                public_straight_type = "double_ended"
                public_has_straight = True
                break
        if not public_has_straight and 12 in public_ranks and 0 in public_ranks and 1 in public_ranks and not (2 in public_ranks and 3 in public_ranks):
            public_straight_type = "inside"
            public_has_straight = True

    if flush_potential and has_straight and dominant_suit in hand_suits:
        return "潜双抽，影响强正面"
    if flush_potential:
        effect = "潜同花"
        strength = "strong_positive" if dominant_suit in hand_suits else "strong_negative"
    elif has_straight:
        effect = "潜顺子"
        strength = {"double_ended": "strong_positive", "inside": "weak_positive"}[straight_type]
    elif public_has_straight:
        effect = "潜顺子"
        hand_overlap = any(rank in set(public_ranks) for rank in hand_ranks)
        strength = {"double_ended": "strong_positive" if hand_overlap else "strong_negative", "inside": "weak_positive" if hand_overlap else "weak_negative"}[public_straight_type]
    elif any(count >= 2 for count in Counter(public_ranks).values()):
        effect = "潜三条"
        strength = "weak_positive" if any(rank in hand_ranks for rank, count in Counter(public_ranks).items() if count >= 2) else "weak_negative"

    return effect if strength == "neutral" else f"{effect}，影响{'正面' if 'positive' in strength else '负面'}"



def format_card(card):
    suits = {'s': '♠', 'h': '♥', 'c': '♣', 'd': '♦'}
    if len(card) >= 2:
        rank = card[:-1]
        suit = card[-1].lower()
        return f"{rank}{suits.get(suit, '')}"
    return card

def calculate_pot_odds(current_pot, bet_to_call):
    if bet_to_call == 0:
        return float('inf')
    elif bet_to_call > 0:
        total_pot = current_pot + bet_to_call
        pot_odds = round(total_pot / bet_to_call, 2)
    else:
        pot_odds = "未知"
    return pot_odds

amd, dmd, xmd, aipa, iden, rule, prefilter_enabled, correction_enabled = Read_Config(type="mode")

def MA_JC(hd, pc, chip, hold, bett, blind, mypos, after, history, pr=3):
    try:
        chip = float(chip) if float(chip) > 0 else float(blind)
    except Exception:
        chip = 1

    try:
        bett_float = float(bett)
    except Exception:
        bett_float = blind

    try:
        hold_float = float(hold)
    except Exception:
        hold_float = 3

    pr = min(pr, 6)
    MM = MA_MM(hd, pc, pr)
    MTKL, PL = MA_PL(hd, pc)
    WIN = round((MM + PL) / 2, 2)
    if isinstance(hd, list):
        hd_symbol = "".join(format_card(card) for card in hd)
    else:
        hd_symbol = "".join(format_card(hd[i:i + 2]) for i in range(0, len(hd), 2))

    if isinstance(pc, list):
        pc_symbol = "".join(format_card(card) for card in pc)
    else:
        pc_symbol = "".join(format_card(pc[i:i + 2]) for i in range(0, len(pc), 2))

    affect = ML_GP(hd, pc)
    pc_count = len(pc) if isinstance(pc, list) else len(pc) // 2
    if pc_count == 0:
        circle, cnum = "底牌圈", 0
    elif pc_count == 3:
        circle, cnum = "翻牌圈", 1
    elif pc_count == 4:
        circle, cnum = "转牌圈", 2
    elif pc_count == 5:
        circle, cnum = "河牌圈", 3
    else:
        circle, cnum = "未知", -1

    if bett_float <= 0:
        bett_float = 0

    odss = calculate_pot_odds(chip, bett_float)
    
    # ===== P0修复：调用辅助函数获取增强信息 =====
    # 1. 计算对手下注尺度（5档）
    bet_percent, bet_category, bet_desc = calculate_bet_size_category(bett_float, chip)
    
    # 2. 分析对手类型（基于历史）
    opponent_type, opponent_desc = analyze_opponent_type(history)
    
    # 3. 分析公牌纹理
    board_texture, board_features = analyze_board_texture(pc)
    
    # 构建增强版描述
    base_desc = (f"当前情景：\n"
                 f"- 手牌：{hd_symbol}\n"
                 f"- 公牌：{pc_symbol}\n"
                 f"- 公牌纹理：{board_texture}\n"
                 f"- 对手下注：{bett_float}（{bet_desc}）\n"
                 f"- 底池大小：{chip}\n"
                 f"- 我的筹码：{hold_float}\n"
                 f"- 剩余对手：{pr}\n"
                 f"- 我的位置：{mypos}\n"
                 f"- 大盲注：{blind}BB\n"
                 f"- 底池赔率：{odss}\n"
                 f"- 后面行动：有{after}个对手\n"
                 f"对手分析：\n"
                 f"- 对手类型：{opponent_type}\n"
                 f"- 对手特征：{opponent_desc}\n"
                 f"- 本手牌局行动历史：{history}\n"
                 f"牌力参数：\n"
                 f"- 蒙特卡洛：{MTKL}（1-7555值，数值越小代表牌力越强）\n"
                 f"决策选项：\n"
                 f"注意，动作和下注量只能从提供的可选项中选择最优项。\n")

    # raise_text = f"- raise（可选betting下注量：{round(chip * 1 / 3, 2)}、{round(chip * 1 / 2,2)}、{round(chip * 2 / 3,2)}、{chip}、{round(chip * 1.2,2)} 或 allin（{hold_float}））\n"
    def Cero(x):
        return math.ceil(x * 100) / 100

    # raise_text = f"- raise（可选betting下注量：底池1/3倍（{Cero(chip / 3)}）、底池1/2倍（{Cero(chip / 2)}）、底池2/3倍（{Cero(chip * 2 / 3)}）、底池1倍（{chip}）、底池1.2倍（{Cero(chip * 1.2)}）、allin）\n"
    raise_text = f"raise（可选betting下注量是底池的：1/3、1/2、2/3、1/1、1.2/1、allin）"
    # 动态生成选项部分
    if bett_float == 0:
        options = (f"1、可选动作：\n"
                   f"- check（可选betting下注量：0）\n"
                   f"- {raise_text}\n")
    elif bett_float >= hold_float:
        options = (f"1、可选动作：\n"
                   f"- call（可选betting下注量：allin {hold_float}）\n"
                   f"- fold（可选betting下注量：0）\n")
    else:
        options = (f"1. 必须遵守最小加注规则（raise至少{bett_float}）。\n"
                   f"2、可选动作：\n"
                   f"- call（可选betting下注量：{bett_float}）\n"
                   f"- {raise_text}\n"
                   f"- fold（可选betting下注量：0）\n")

    if bett_float >= hold_float * 0.5:
        committed_percent = (bett_float / hold_float) * 100 if hold_float > 0 else 0
        options += f"\n⚠️ 注意：面临commitment决策点（已投入{committed_percent:.0f}），Commitment处理原则：\n" \
                   "- 有摊牌价值：直接allin\n" \
                   "- 听牌：计算隐含赔率后决策\n" \
                   "- 空气牌：立即弃牌"

    endtext = ("\n输出规则：\n"
               '- 返回纯净JSON格式：{"action": "<动作>", "betting": "<下注量>"}\n'
               '- action：动作\n'
               '- betting：下注量（0表示不下注）')

    # 合并最终描述
    describe = base_desc + options + endtext
    BoxText("m")

    # ===== 三层决策机制（优化版） =====
    
    # 【第1层：预过滤层】- 只过滤极垃圾牌（可配置开关）
    if prefilter_enabled and WIN <= float(rule) and cnum == 0:
        if bett_float <= 0:
            action = "Check"
            betting = "0"
            BoxText(f"预过滤：胜率{WIN}%≤{rule}%，免费看牌Check")
        else:
            action = "Fold"
            betting = "0"
            BoxText(f"预过滤：胜率{WIN}%≤{rule}%，弃牌保护资金")
            # 语音播报决策（英文）
            speak_decision(action)
    else:
        # 【第2层：AI决策层】- 主要决策（信任AI）
        Xarr = Good_Idea(describe, cnum)
        Xarr = Str_Json(Xarr)
        action = Xarr['action']
        betting = Xarr['betting']
        BoxText(f"AI决策：{action} | {betting}")
        
        # 语音播报决策（英文）
        speak_decision(action)

        # 【第3层：纠错层】- 只修正违反游戏规则的错误（可配置开关）
        corrected = False
        if correction_enabled:
            # 纠正1：免费看牌时不能弃牌
            if action.lower() == "fold" and bett_float == 0:
                action = "Check"
                betting = "0"
                BoxText(f"纠错：免费看牌改为Check")
                corrected = True
            
            # 纠正2：底牌圈极高胜率保护（只在胜率>85%时干预）
            elif action == "fold" and WIN >= 85 and cnum == 0 and bett_float <= blind * 2:
                action = "Call"
                betting = str(bett_float)
                BoxText(f"纠错：胜率{WIN}%极高，强制跟注")
                corrected = True
        else:
            # 不启用纠错时，只做基本规则检查
            if action.lower() == "fold" and bett_float == 0:
                action = "Check"
                betting = "0"
                BoxText(f"规则检查：免费看牌改为Check")
                corrected = True
        
        # 如果决策被纠正，重新播报（英文）
        if corrected:
            speak_decision(action)

    lines = [f"行动：{action}"]
    if action.lower() in ["raise", "call"]:
        lines.append(f"下注：{betting}")

    ap = aipa[cnum] if 0 <= cnum < len(aipa) else aipa
    lines.extend([f"手牌：{hd_symbol}", f"公牌：{pc_symbol}", f"对手：{pr}", f"位置：{mypos}", f"锅底：{chip}", f"筹码：{hold_float}", f"低消：{bett_float}", f"大盲：{blind}", f"胜率：{WIN}", f"影响：{affect}",
                  f"引擎：{ap}"])
    OutText = "\n".join(lines)
    return action, betting, WIN, OutText


# ===== P0修复：辅助函数 =====

def calculate_bet_size_category(bett_float, chip):
    """计算对手下注尺度分类（5档详细版，容错增强）"""
    try:
        # 容错处理：参数异常
        if bett_float is None or chip is None:
            return 0, "信息缺失", "下注信息未识别"
        
        bett_float = float(bett_float) if bett_float else 0
        chip = float(chip) if chip else 0
        
        if bett_float <= 0 or chip <= 0:
            return 0, "无下注", "check/让牌"
        
        # 计算下注占底池的倍数
        bet_ratio = bett_float / chip
        bet_percent = round(bet_ratio * 100, 1)
        
        # 5档分类
        if bet_ratio <= 1/3:
            category = "微注"
            desc = f"{bet_percent}%底池（≤1/3，微注）"
        elif bet_ratio <= 1/2:
            category = "小注"
            desc = f"{bet_percent}%底池（≤1/2，小注）"
        elif bet_ratio <= 2/3:
            category = "标准注"
            desc = f"{bet_percent}%底池（≤2/3，标准注）"
        elif bet_ratio <= 1.0:
            category = "中注"
            desc = f"{bet_percent}%底池（≤1倍，中注）"
        else:
            category = "大注" if bet_ratio < 1.5 else "超额下注"
            desc = f"{bet_percent}%底池（{bet_ratio:.1f}倍，{category}）"
        
        return bet_percent, category, desc
    except Exception as e:
        # 异常兜底：返回安全默认值
        return 50.0, "标准注", "50%底池（信息异常，按标准注处理）"


def analyze_opponent_type(history):
    """基于历史行动推断对手类型（简化版，容错增强）"""
    try:
        # 容错处理：无历史或信息不足
        if not history or history is None:
            return "标准型", "无历史数据（按GTO平衡策略应对）"
        
        # 转换为字符串处理
        history = str(history)
        
        if len(history) < 10:
            return "标准型", "数据不足（按GTO平衡策略应对）"
        
        history_lower = history.lower()
        
        # 统计关键行为
        raise_count = history.count("加注") + history.count("下注")
        call_count = history.count("跟注")
        check_count = history.count("让牌") + history_lower.count("check")
        fold_count = history.count("弃牌")
        
        total_actions = raise_count + call_count + check_count + fold_count
        if total_actions == 0:
            return "标准型", "无有效行动（按GTO平衡策略应对）"
        
        # 计算比率
        aggression_rate = raise_count / total_actions if total_actions > 0 else 0
        passive_rate = (call_count + check_count) / total_actions if total_actions > 0 else 0
        
        # 判断类型
        if aggression_rate >= 0.5:
            if fold_count > raise_count * 0.3:
                opponent_type = "紧凶（TAG）"
                desc = "入池少但激进，尊重其下注"
            else:
                opponent_type = "松凶（LAG）"
                desc = "频繁加注，用强牌陷阱"
        elif passive_rate >= 0.6:
            if call_count > check_count:
                opponent_type = "松被动（LP）"
                desc = "爱跟注，加大价值注"
            else:
                opponent_type = "紧弱（TP）"
                desc = "很被动，可激进偷池"
        else:
            opponent_type = "标准型"
            desc = "平衡打法，按GTO决策"
        
        # 添加具体数据
        detail = f"（激进度{aggression_rate*100:.0f}%，被动度{passive_rate*100:.0f}%）"
        
        return opponent_type, desc + detail
    except Exception as e:
        # 异常兜底：返回安全默认值
        return "标准型", "信息异常（按GTO平衡策略应对）"


def analyze_board_texture(pc):
    """分析公牌纹理（详细版，容错增强）"""
    try:
        # 容错处理：无公牌或信息缺失
        if not pc or pc is None or len(pc) == 0:
            return "底牌圈（无公牌）", []
        
        # 转换为列表
        if not isinstance(pc, list):
            pc = [pc[i:i+2] for i in range(0, len(pc), 2)]
        
        if len(pc) < 3:
            return "公牌不足（按中等危险度处理）", []
        
        # 提取花色和点数
        suits = [card[-1].lower() for card in pc if len(card) >= 2]
        ranks = [card[:-1].upper() for card in pc if len(card) >= 2]
        
        # 点数映射
        rank_values = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
        rank_nums = sorted([rank_values.get(r, 0) for r in ranks if r in rank_values])
        
        features = []
        danger_level = 0  # 危险等级 0-10
        
        # 1. 花色分析
        suit_count = {}
        for suit in suits:
            suit_count[suit] = suit_count.get(suit, 0) + 1
        
        max_suit_count = max(suit_count.values()) if suit_count else 0
        
        if max_suit_count >= 4:
            features.append("四同花")
            danger_level += 4
        elif max_suit_count == 3:
            features.append("三同花面")
            danger_level += 3
        elif max_suit_count == 2:
            features.append("两同花")
            danger_level += 1
        else:
            features.append("彩虹面")
        
        # 2. 顺子面分析
        if len(rank_nums) >= 3:
            # 检查连续性
            max_connected = 1
            current_connected = 1
            for i in range(len(rank_nums)-1):
                if rank_nums[i+1] - rank_nums[i] <= 2:
                    current_connected += 1
                    max_connected = max(max_connected, current_connected)
                else:
                    current_connected = 1
            
            if max_connected >= 4:
                features.append("四顺子面")
                danger_level += 3
            elif max_connected == 3:
                if rank_nums[-1] - rank_nums[0] <= 4:
                    features.append("三连顺子面")
                    danger_level += 2
                else:
                    features.append("有缺口顺子面")
                    danger_level += 1
        
        # 3. 对子/三条分析
        rank_count = {}
        for rank in ranks:
            rank_count[rank] = rank_count.get(rank, 0) + 1
        
        max_rank_count = max(rank_count.values()) if rank_count else 1
        
        if max_rank_count == 3:
            features.append("三条")
            danger_level += 3
        elif max_rank_count == 2:
            pair_count = sum(1 for v in rank_count.values() if v == 2)
            if pair_count >= 2:
                features.append("两对")
                danger_level += 2
            else:
                features.append("一对")
                danger_level += 1
        
        # 4. 高牌分析
        high_cards = [r for r in ranks if r in ['A', 'K', 'Q', 'J']]
        if len(high_cards) >= 3:
            features.append("三高牌")
            danger_level += 1
        elif len(high_cards) >= 2:
            features.append("两高牌")
        elif len(high_cards) >= 1:
            features.append("单高牌")
        else:
            features.append("低牌面")
            danger_level -= 1
        
        # 综合判断纹理
        if danger_level >= 6:
            texture = "极湿润公牌"
        elif danger_level >= 4:
            texture = "湿润公牌"
        elif danger_level >= 2:
            texture = "中等公牌"
        else:
            texture = "干燥公牌"
        
        feature_str = "、".join(features)
        full_desc = f"{texture}（{feature_str}，危险度{danger_level}）"
        
        return full_desc, features
    except Exception as e:
        # 异常兜底：返回安全默认值
        return "中等公牌（信息异常，按中等危险度处理）", []
